from spl_ft.benchmark import summarize,harbor_config
from spl_ft.config import Benchmark
from spl_ft.inference import decode_output,normalize_messages


def test_infrastructure_errors_do_not_shrink_denominator():
    records=[{'task_name':'a','verifier_result':{'rewards':{'reward':1}},'reward_valid':True},
             {'task_name':'b','exception_info':{'exception_type':'BuildError'}},
             {'task_name':'c','verifier_result':{'rewards':{'reward':0}},'reward_valid':True}]
    s=summarize(records,4)
    assert s['success_rate']==.25
    assert s['graded_success_rate']==.5
    assert not s['complete']
    assert s['errors']==1 and s['graded']==2


def test_false_reward_valid_cannot_count_as_pass():
    s=summarize([{'task_name':'a','reward_valid':False,'verifier_result':{'rewards':{'reward':1}}}],1)
    assert s['passed']==0 and not s['complete']


def test_harbor_agent_and_network_config(tmp_path):
    c=harbor_config(Benchmark(agent='opencode',architecture='amd64'),tmp_path,[{'path':'/tmp/task'}],'http://host.docker.internal:123/v1',[],agent_version='1.2.3')
    assert c['agents'][0]['base_url'].startswith('http://host.docker.internal')
    assert 'opencode_config' in c['agents'][0]['kwargs']
    assert c['agents'][0]['kwargs']['version']=='1.2.3'
    assert c['environment']['extra_docker_compose']
    assert 'linux/amd64' in (tmp_path/'network.json').read_text()


def test_lfm3_reasoning_and_typed_tool_call_parser():
    tools=[{'type':'function','function':{'name':'bash','parameters':{'properties':{'command':{'type':'string'},'timeout':{'type':'integer'}}}}}]
    out=decode_output('think first</think><tool_call>\n<function=bash>\n<parameter=command>\npython x.py\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>',tools)
    assert out['reasoning_content']=='think first'
    assert out['tool_calls'][0]['function']['name']=='bash'
    import json
    assert json.loads(out['tool_calls'][0]['function']['arguments'])=={'command':'python x.py','timeout':30}
    assert normalize_messages([out])[0]['tool_calls'][0]['function']['arguments']['timeout']==30


def test_unfinished_reasoning_not_fabricated_final_answer():
    out=decode_output('still thinking')
    assert not out['content'] and out['reasoning_content']=='still thinking'
