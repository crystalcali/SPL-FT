import json
import os
import pytest
from spl_ft.config import Experiment
from spl_ft.data import normalize, Pools, windows, tokenize_row


def row(i, variant='answer'):
    return normalize({'conversations':[{'from':'human','value':f'Task {i}'},{'from':'gpt','value':variant}]})


def sample(path, n, rows):
    p=Pools(path,Experiment(train_size=n,test_size=4,val_size=3))
    for r in rows:p.add(r)
    return p.finish()


def test_holdout_independent_of_train_size_order_and_duplicate_variant(tmp_path):
    rows=[row(i) for i in range(100)]+[row(i,'another answer') for i in range(30)]
    a=sample(tmp_path/'a.db',5,rows)
    b=sample(tmp_path/'b.db',30,list(reversed(rows)))
    for split in ('test','validation'):
        assert a[split]==b[split]
    groups=[{r['group'] for r in a[s]} for s in ('train','validation','test')]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    assert {r['id'] for r in a['train']} <= {r['id'] for r in b['train']}


def test_td1_terminal_screen_not_part_of_group():
    def make(screen):
        return normalize({'conversations':[{'role':'user','content':f'Instructions\nTask Description:\nFix the project\nCurrent terminal state:\n{screen}'},{'role':'assistant','content':'done'}]})
    assert make('host A')['group']==make('host B')['group']
    assert make('host A')['messages']!=make('host B')['messages']


def test_insufficient_unique_tasks_is_error(tmp_path):
    with pytest.raises(ValueError,match='requested'):
        sample(tmp_path/'a.db',5,[row(1)]*50)


class Tokenizer:
    def apply_chat_template(self,messages,tokenize=True,return_dict=False,**kw):
        ids=[];masks=[]
        for m in messages:
            prefix=[ord(x) for x in '<'+m['role']+'>']
            text=[ord(x) for x in m['content']+'!']
            ids+=prefix+text;masks += [0]*len(prefix)+[m['role']=='assistant']*len(text)
        return {'input_ids':ids,'assistant_masks':masks} if return_dict else ids


def test_native_masks_never_train_observations():
    t=Tokenizer();r=normalize({'messages':[{'role':'user','content':'TASK'},{'role':'assistant','content':'AAA'},{'role':'tool','content':'OBS'},{'role':'assistant','content':'BBB'}]})
    z=tokenize_row(t,r)
    text=''.join(chr(v) for v in z['labels'] if v!=-100)
    assert text=='AAA!BBB!'


def test_windows_keep_task_and_supervise_targets_once():
    t=Tokenizer();messages=[{'role':'user','content':'TASK'}]
    for i in range(5):
        messages += [{'role':'assistant','content':chr(65+i)*4},{'role':'tool','content':'observation'*3}]
    ws,d=windows(t,normalize({'messages':messages}),70)
    assert d==0
    assert len(ws)==5
    targets=[]
    for w in ws:
        text=''.join(chr(v) for v in w['input_ids'])
        assert 'TASK' in text
        assert len(w['input_ids'])<=70
        targets.append(''.join(chr(v) for v in w['labels'] if v!=-100))
    assert targets==[chr(65+i)*4+'!' for i in range(5)]


def test_oversize_answer_dropped_not_truncated():
    r=normalize({'messages':[{'role':'user','content':'TASK'},{'role':'assistant','content':'A'*200}]})
    ws,d=windows(Tokenizer(),r,50)
    assert ws==[] and d==1


@pytest.mark.skipif(not os.getenv('SPL_FT_TOKENIZER_PATH'),reason='Opt-in real local LFM3 tokenizer')
def test_real_lfm3_native_template_masks_and_windows():
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(os.environ['SPL_FT_TOKENIZER_PATH'],trust_remote_code=True)
    r=normalize({'messages':[{'role':'user','content':'INITIAL_TASK'},{'role':'assistant','content':'<think>FIRST_REASONING</think>FIRST_RESULT'},
        {'role':'user','content':'OBSERVATION_ONLY '*500},{'role':'assistant','content':'<think>LAST_REASONING</think>LAST_RESULT'}]})
    all_labels=tokenize_row(t,r)['labels']
    decoded=t.decode([v for v in all_labels if v!=-100])
    assert 'OBSERVATION_ONLY' not in decoded
    assert 'FIRST_RESULT' in decoded and 'LAST_RESULT' in decoded
    ws,d=windows(t,r,256)
    assert d==0
    assert len(ws)==2
    assert all(len(w['input_ids'])<=256 for w in ws)
    targets=[t.decode([v for v in w['labels'] if v!=-100]) for w in ws]
    assert 'FIRST_RESULT' in targets[0] and 'LAST_RESULT' not in targets[0]
    assert 'LAST_RESULT' in targets[1] and 'FIRST_RESULT' not in targets[1]
