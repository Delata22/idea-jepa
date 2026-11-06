"""Object state at a bAbI qa2 question, shared by babi_dropcase.py and babi_scale_report.py."""
MOVE = {"moved", "went", "journeyed", "travelled"}
TAKE = {"got", "picked", "grabbed", "took"}
DROP = {"dropped", "left", "discarded", "put"}


def analyse(st, obj):
    pos, holder, last_act, carrier, moved_after_drop = {}, None, None, None, False
    events = []
    for i, s in enumerate(st):
        w = [x for x in s if x not in (".", "there", "back", "up", "down")]
        who, verb, o = w[0], w[1], w[-1]
        if verb in MOVE:
            pos[who] = o
            if holder == who:
                last_act = i
            elif carrier == who and holder is None:
                moved_after_drop = True
            events.append((i, who))
        elif verb in TAKE and o == obj:
            holder, carrier, last_act, moved_after_drop = who, who, i, False
        elif verb in DROP and o == obj:
            holder, last_act, moved_after_drop = None, i, False
    if carrier is None:
        return None
    case = "a_carried" if holder is not None else ("c_dropped_carrier_left" if moved_after_drop else "b_dropped_carrier_stayed")
    others = sum(1 for i, who in events if i > last_act and who != carrier)
    return case, others, pos.get(carrier)
