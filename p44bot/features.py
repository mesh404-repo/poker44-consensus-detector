"""
Poker44 bot-detection feature extraction (SHARED between training and the live miner).

Input: `hands` = one chunk = list of miner-visible hand dicts. Each chunk features the
same focus player at metadata.hero_seat across ~30-40 hands; the label is whether that
focus player is a bot.

Design: for every hand we compute ~45 per-hand scalar metrics (table context + hero
behaviour). We aggregate each across the chunk with 7 statistics (mean/std/min/max/
q10/q50/q90). We add sequence-repetition "signature" features (a strong bot tell) and
hero-centric rate features (VPIP/PFR/response-to-aggression) that competitors lack.

Only miner-visible fields are used (no cards/outcome/showdown/board/button — always empty
live). Pure python + numpy so the live miner stays light.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

BB_UNIT = 0.02
ACTION_KINDS = ("fold", "check", "call", "bet", "raise")
ROUND_BB_STACKS = (20, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500)


def _subsample_actions(actions, k=8):
    """Match the LIVE payload, which subsamples each hand to <=8 actions (keep first, last,
    evenly-spaced middle). No-op on live data (already <=8); on benchmark (up to 23) it aligns
    the action distribution so train == serve."""
    n = len(actions)
    if n <= k:
        return actions
    idx = {0, n - 1}
    step = (n - 1) / (k - 1)
    for i in range(1, k - 1):
        idx.add(int(round(i * step)))
    return [actions[i] for i in sorted(idx)[:k]]


def _f(x, d=0.0) -> float:
    try:
        v = float(x)
        return d if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return d


def _i(x, d=0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return d


def _div(a, b) -> float:
    return float(a) / float(b) if b else 0.0


def _mean(v: List[float]) -> float:
    return _div(sum(v), len(v))


def _std(v: List[float]) -> float:
    if not v:
        return 0.0
    m = _mean(v)
    return math.sqrt(max(0.0, _mean([(x - m) ** 2 for x in v])))


def _quant(v: List[float], q: float) -> float:
    if not v:
        return 0.0
    xs = sorted(v)
    if len(xs) == 1:
        return xs[0]
    pos = min(max(q, 0.0), 1.0) * (len(xs) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] if lo == hi else xs[lo] * (1 - (pos - lo)) + xs[hi] * (pos - lo)


def _norm_entropy(items) -> float:
    c = Counter(items)
    tot = float(sum(c.values()))
    if tot <= 0 or len(c) <= 1:
        return 0.0
    ent = -sum((n / tot) * math.log(n / tot + 1e-12) for n in c.values())
    return _div(ent, math.log(len(c)))


def _max_run_share(seq) -> float:
    if not seq:
        return 0.0
    longest = cur = 1
    for a, b in zip(seq, seq[1:]):
        cur = cur + 1 if a == b else 1
        longest = max(longest, cur)
    return _div(longest, len(seq))


def _amt_bucket(v: float) -> str:
    if v <= 0.0:
        return "z"
    if v <= 0.5:
        return "xs"
    if v <= 1.0:
        return "s"
    if v <= 2.0:
        return "m"
    if v <= 3.0:
        return "ml"
    if v <= 5.0:
        return "l"
    if v <= 8.0:
        return "xl"
    return "xxl"


def _hand(hand: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Return (per-hand scalar features, aux sequences/hero counters)."""
    md = hand.get("metadata") or {}
    hero = _i(md.get("hero_seat"), 0)
    max_seats = max(1, _i(md.get("max_seats"), 6))
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = _subsample_actions(hand.get("actions") or [])  # match live <=8 actions/hand

    stacks_bb, hero_stack_bb = [], 0.0
    for p in players:
        if isinstance(p, dict):
            s = _div(_f(p.get("starting_stack")), BB_UNIT)
            stacks_bb.append(s)
            if _i(p.get("seat")) == hero:
                hero_stack_bb = s
    # magnitude-invariant stacks: live hero ~100bb (constant) vs benchmark ~230bb (varied).
    # Rescale so hero=100bb, preserving relative ratios -> matches live, keeps opponent-relative signal.
    # The SAME factor is applied to pot/bet amounts below (amt/pb/pa) so every magnitude feature is in
    # consistent hero=100bb units -> kills the deep-stack benchmark->live shift on pot_growth/amount_bb
    # (the top residual transfer liabilities); ratio features (bet/pot) are invariant since _sc cancels.
    _sc = 1.0
    if hero_stack_bb > 0:
        _sc = 100.0 / hero_stack_bb
        stacks_bb = [s * _sc for s in stacks_bb]
        hero_stack_bb = 100.0

    a_types, actor_seq, street_seq, amts, pot_b, pot_a = [], [], [], [], [], []
    raise_to_n = call_to_n = 0
    # hero within-hand tallies
    h_cnt = Counter()
    h_bet_bb, h_bet_potfrac = [], []
    faced = fold_f = call_f = raise_f = 0
    vpip = pfr = 0
    street_aggr = {}
    for a in actions:
        if not isinstance(a, dict):
            continue
        at = str(a.get("action_type") or "").lower().strip()
        seat = _i(a.get("actor_seat"))
        st = str(a.get("street") or "").lower().strip()
        amt = max(0.0, _f(a.get("normalized_amount_bb"))) * _sc
        pb = _div(_f(a.get("pot_before")), BB_UNIT) * _sc
        pa = _div(_f(a.get("pot_after")), BB_UNIT) * _sc
        a_types.append(at)
        if seat > 0:
            actor_seq.append(seat)
        street_seq.append(st)
        amts.append(amt)
        pot_b.append(pb)
        pot_a.append(pa)
        raise_to_n += int(a.get("raise_to") is not None)
        call_to_n += int(a.get("call_to") is not None)
        aggr = at in ("bet", "raise")
        if seat == hero and hero > 0 and at in ACTION_KINDS:
            h_cnt[at] += 1
            if street_aggr.get(st):
                faced += 1
                if at == "fold":
                    fold_f += 1
                elif at == "call":
                    call_f += 1
                elif at in ("bet", "raise"):
                    raise_f += 1
            if st == "preflop":
                if at in ("call", "bet", "raise"):
                    vpip = 1
                if at == "raise":
                    pfr = 1
            if aggr:
                h_bet_bb.append(amt)
                if pb > 0:
                    h_bet_potfrac.append(_div(_div(_f(a.get("amount")), BB_UNIT), pb))
        if aggr:
            street_aggr[st] = True

    cnt = Counter(a_types)
    nact = max(1.0, float(len(a_types)))
    meaningful = max(1, sum(cnt.get(k, 0) for k in ACTION_KINDS))
    aggressive = cnt.get("bet", 0) + cnt.get("raise", 0)
    passive = cnt.get("call", 0) + cnt.get("check", 0)
    preflop_n = sum(1 for s in street_seq if s == "preflop")
    postflop_n = sum(1 for s in street_seq if s not in ("", "preflop"))
    pot_delta = [max(0.0, x - y) for x, y in zip(pot_a, pot_b)]
    monotonic = sum(1 for x, y in zip(pot_a, pot_a[1:]) if y + 1e-9 >= x)
    n_streets = len(set(s for s in street_seq if s)) or len(streets)
    hero_actions = h_cnt.get("fold", 0) + h_cnt.get("check", 0) + h_cnt.get("call", 0) + h_cnt.get("bet", 0) + h_cnt.get("raise", 0)

    scal = {
        "player_count": float(len(players)),
        "seat_util": _div(len(players), max_seats),
        "action_count": float(len(a_types)),
        "street_count": float(n_streets),
        "call_share": _div(cnt.get("call", 0), meaningful),
        "check_share": _div(cnt.get("check", 0), meaningful),
        "fold_share": _div(cnt.get("fold", 0), meaningful),
        "bet_share": _div(cnt.get("bet", 0), meaningful),
        "raise_share": _div(cnt.get("raise", 0), meaningful),
        "aggr_share": _div(aggressive, nact),
        "passive_share": _div(passive, nact),
        "preflop_share": _div(preflop_n, nact),
        "postflop_share": _div(postflop_n, nact),
        "action_entropy": _norm_entropy(a_types),
        "actor_entropy": _norm_entropy(actor_seq),
        "street_entropy": _norm_entropy(street_seq),
        "unique_actor_share": _div(len(set(actor_seq)), max(1.0, len(players))),
        "actor_switch_rate": _div(sum(1 for a, b in zip(actor_seq, actor_seq[1:]) if a != b), max(len(actor_seq) - 1, 1)),
        "actor_run_max_share": _max_run_share(actor_seq),
        "action_run_max_share": _max_run_share(a_types),
        "amount_mean_bb": _mean(amts),
        "amount_std_bb": _std(amts),
        "amount_q90_bb": _quant(amts, 0.9),
        "amount_max_bb": max(amts) if amts else 0.0,
        "nonzero_amount_share": _div(sum(1 for v in amts if v > 0), nact),
        "pot_before_mean_bb": _mean(pot_b),
        "pot_after_mean_bb": _mean(pot_a),
        "pot_delta_mean_bb": _mean(pot_delta),
        "pot_growth_bb": (max(pot_a) - min(pot_b)) if (pot_a and pot_b) else 0.0,
        "pot_monotonic_rate": _div(monotonic, max(len(pot_a) - 1, 1)),
        "raise_to_share": _div(raise_to_n, nact),
        "call_to_share": _div(call_to_n, nact),
        "stack_mean_bb": _mean(stacks_bb),
        "stack_std_bb": _std(stacks_bb),
        "stack_iqr_bb": _quant(stacks_bb, 0.75) - _quant(stacks_bb, 0.25),
        "hero_stack_bb": hero_stack_bb,
        "hero_action_share": _div(hero_actions, nact),
        "hero_aggr_in_hand": _div(h_cnt.get("bet", 0) + h_cnt.get("raise", 0), max(1, hero_actions)),
        "hero_fold_in_hand": _div(h_cnt.get("fold", 0), max(1, hero_actions)),
        "reached_flop": 1.0 if n_streets >= 2 else 0.0,
        "reached_turn": 1.0 if n_streets >= 3 else 0.0,
        "reached_river": 1.0 if n_streets >= 4 else 0.0,
        "hero_seat": float(hero),
    }
    aux = {
        "action_sig": tuple(a_types),
        "actor_sig": tuple(actor_seq),
        "street_sig": tuple(street_seq),
        "amt_bucket_sig": tuple(_amt_bucket(v) for v in amts),
        "h_cnt": h_cnt, "faced": faced, "fold_f": fold_f, "call_f": call_f, "raise_f": raise_f,
        "vpip": vpip, "pfr": pfr, "h_bet_bb": h_bet_bb, "h_bet_potfrac": h_bet_potfrac,
        "hero_actions": hero_actions, "hero_stack_bb": hero_stack_bb,
    }
    return scal, aux


def _agg7(out: Dict[str, float], prefix: str, vals: List[float]) -> None:
    out[prefix + "_mean"] = _mean(vals)
    out[prefix + "_std"] = _std(vals)
    out[prefix + "_min"] = min(vals) if vals else 0.0
    out[prefix + "_max"] = max(vals) if vals else 0.0
    out[prefix + "_q10"] = _quant(vals, 0.1)
    out[prefix + "_q50"] = _quant(vals, 0.5)
    out[prefix + "_q90"] = _quant(vals, 0.9)


def extract_features(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    hands = [h for h in hands if isinstance(h, dict)]
    if not hands:
        return {"hand_count": 0.0}
    scal, aux = [], []
    for h in hands:
        s, a = _hand(h)
        scal.append(s)
        aux.append(a)
    n = float(len(hands))
    out: Dict[str, float] = {"hand_count": n}

    # ---- 7-stat aggregation of every per-hand scalar ----
    for name in scal[0].keys():
        _agg7(out, name, [s[name] for s in scal])

    # ---- sequence-repetition signatures (bot tell) ----
    for sig in ("action_sig", "actor_sig", "street_sig", "amt_bucket_sig"):
        seqs = [a[sig] for a in aux]
        c = Counter(seqs)
        out[sig + "_top_share"] = _div(max(c.values()), n)
        out[sig + "_uniq_share"] = _div(len(c), n)

    # ---- special-hand rates ----
    out["rate_high_aggr_hand"] = _div(sum(1 for s in scal if s["aggr_share"] >= 0.35), n)
    out["rate_low_action_entropy_hand"] = _div(sum(1 for s in scal if s["action_entropy"] <= 0.35), n)
    out["rate_high_actor_entropy_hand"] = _div(sum(1 for s in scal if s["actor_entropy"] >= 0.75), n)
    out["rate_long_action_hand"] = _div(sum(1 for s in scal if s["action_count"] >= 12.0), n)
    out["rate_hero_vpip_hand"] = _div(sum(a["vpip"] for a in aux), n)
    out["rate_hero_pfr_hand"] = _div(sum(a["pfr"] for a in aux), n)

    # ---- HERO-CENTRIC chunk rates (my differentiator) ----
    tot_hero = sum(a["hero_actions"] for a in aux)
    for k in ACTION_KINDS:
        out["hero_%s_rate" % k] = _div(sum(a["h_cnt"].get(k, 0) for a in aux), tot_hero)
    tot_aggr = sum(a["h_cnt"].get("bet", 0) + a["h_cnt"].get("raise", 0) for a in aux)
    out["hero_aggr_rate"] = _div(tot_aggr, tot_hero)
    out["hero_aggr_factor"] = _div(tot_aggr, sum(a["h_cnt"].get("call", 0) for a in aux) or 1)
    out["hero_action_entropy"] = _norm_entropy(
        {k: sum(a["h_cnt"].get(k, 0) for a in aux) for k in ACTION_KINDS})
    out["hero_actions_per_hand_mean"] = _div(tot_hero, n)
    out["hero_actions_per_hand_std"] = _std([float(a["hero_actions"]) for a in aux])
    out["vpip"] = _div(sum(a["vpip"] for a in aux), n)
    out["pfr"] = _div(sum(a["pfr"] for a in aux), n)
    out["vpip_pfr_gap"] = out["vpip"] - out["pfr"]
    faced = sum(a["faced"] for a in aux)
    out["fold_to_aggr"] = _div(sum(a["fold_f"] for a in aux), faced)
    out["call_to_aggr"] = _div(sum(a["call_f"] for a in aux), faced)
    out["raise_to_aggr"] = _div(sum(a["raise_f"] for a in aux), faced)
    out["faced_aggr_rate"] = _div(faced, tot_hero)

    # hero bet sizing + regularity
    all_bet = [x for a in aux for x in a["h_bet_bb"]]
    all_pf = [x for a in aux for x in a["h_bet_potfrac"]]
    _agg7(out, "hero_betbb", all_bet)
    _agg7(out, "hero_potfrac", all_pf)
    out["hero_betbb_cv"] = _div(out["hero_betbb_std"], out["hero_betbb_mean"] or 1.0)
    out["hero_bet_uniq_ratio"] = _div(len(set(round(x, 1) for x in all_bet)), len(all_bet) or 1)
    out["hero_n_bets_per_hand"] = _div(len(all_bet), n)
    if all_bet:
        modal = Counter(round(x, 1) for x in all_bet).most_common(1)[0][1]
        out["hero_bet_modal_share"] = _div(modal, len(all_bet))
    else:
        out["hero_bet_modal_share"] = 0.0

    # hero stack tells
    hstacks = [a["hero_stack_bb"] for a in aux if a["hero_stack_bb"] > 0]
    out["hero_stack_round_share"] = _div(
        sum(1 for s in hstacks if min(abs(s - r) for r in ROUND_BB_STACKS) <= 1.0), len(hstacks) or 1)
    out["hero_stack_uniq_ratio"] = _div(len(set(round(s) for s in hstacks)), len(hstacks) or 1)

    return out


FEATURE_VERSION = "v2"

if __name__ == "__main__":
    import json, sys
    ex = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "/root/sn126/data/examples.jsonl")]
    f = extract_features(ex[0]["hands"])
    print("nfeat =", len(f), "| label =", ex[0]["label"])
