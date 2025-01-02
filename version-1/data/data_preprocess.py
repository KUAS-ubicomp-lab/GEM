import json
from tqdm import tqdm


def get_relation(ctx_spk_index, ctx_adr_index, rsp_adr_index):
    ctx_relation = []
    ctx_spk_2_utr = {}
    for utr_id, (utr_spk, utr_adr) in enumerate(zip(ctx_spk_index, ctx_adr_index)):
        if utr_spk in ctx_spk_2_utr:
            ctx_spk_2_utr[utr_spk].append(utr_id)
        else:
            ctx_spk_2_utr[utr_spk] = [utr_id]

        if utr_adr == -1:
            continue

        if utr_adr in ctx_spk_2_utr:
            src = ctx_spk_2_utr[utr_adr][-1]
            tgt = utr_id
            ctx_relation.append([tgt, src])

    if rsp_adr_index in ctx_spk_2_utr:
        rsp_idx = ctx_spk_2_utr[rsp_adr_index][-1]
    else:
        rsp_idx = -1

    return ctx_relation, rsp_idx


# main
for mpc_len in [5, 10, 15]:
    print("Processing the dataset of conversation length: {} ...".format(mpc_len))

    with open("{}.json".format(mpc_len), "r") as fin:
        data = json.load(fin)

        for split, dialogues in data.items():
            with open("{}_{}.json".format(mpc_len, split), "w") as fout:

                for dialogue in tqdm(dialogues, total=len(dialogues)):
                    assert len(dialogue) == mpc_len

                    user_index = {'-': -1}

                    # context
                    ctx = []
                    ctx_spk = []
                    for utterance in dialogue[:-1]:
                        assert len(utterance) == 3
                        utr_spk = utterance[0]
                        utr = utterance[1]
                        utr_adr = utterance[2]

                        if utr_spk not in user_index:
                            user_index[utr_spk] = len(user_index)
                        if utr_adr not in user_index:
                            user_index[utr_adr] = len(user_index)

                        ctx.append(utr)
                        ctx_spk.append(user_index[utr_spk])

                    # response
                    response = dialogue[-1]
                    rsp_spk = response[0]
                    rsp = response[1]
                    rsp_adr = response[2]

                    if rsp_spk not in user_index:
                        user_index[rsp_spk] = len(user_index)
                    assert rsp_adr in user_index
                    assert rsp_adr != '-'

                    rsp_spk = user_index[rsp_spk]
                    rsp_adr = user_index[rsp_adr]

                    ctx_relation, rsp_idx = get_relation(ctx_spk, rsp_adr)

                    d = {'context': ctx, 'relation_at': ctx_relation, 'ctx_spk': ctx_spk,
                         'label': rsp, 'ans_idx': rsp_idx, 'ans_spk': rsp_spk, 'ans_adr': rsp_adr}

                    json_str = json.dumps(d)
                    fout.write(json_str + '\n')
