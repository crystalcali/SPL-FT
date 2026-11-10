import io
import zipfile
from fastapi.testclient import TestClient
from spl_ft.app import create_app,access_token


def client(tmp_path):
    app=create_app(tmp_path,start_worker=False)
    c=TestClient(app)
    assert c.get('/api/state').status_code==401
    assert c.post('/api/login',json={'token':access_token(tmp_path)}).status_code==200
    return c


def test_auth_csrf_queue_and_schema(tmp_path):
    c=client(tmp_path)
    assert c.post('/api/experiments',json={'method':'unknown'}).status_code==422
    assert c.post('/api/experiments',json={},headers={'origin':'http://evil.example'}).status_code==403
    j=c.post('/api/experiments',json={'auto_benchmark':True}).json()
    jobs=c.get('/api/state').json()['jobs']
    assert len(jobs)==2
    assert next(x for x in jobs if x['kind']=='benchmark')['dependency']==j['id']
    assert c.post('/api/jobs/'+j['id']+'/cancel').json()['state']=='cancelled'
    assert c.get('/api/jobs/unknown').status_code==404
    assert c.post('/api/logout',json={}).status_code==200
    assert c.get('/api/state').status_code==401


def test_skill_upload_and_unsafe_archive(tmp_path):
    c=client(tmp_path)
    r=c.post('/api/skills',files={'file':('SKILL.md',b'# general workflow')})
    assert r.status_code==200
    assert len(c.get('/api/skills').json())==1
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:z.writestr('../SKILL.md','unsafe')
    assert c.post('/api/skills',files={'file':('bad.zip',stream.getvalue())}).status_code==400
    assert not (tmp_path/'SKILL.md').exists()


def test_retry_resume_requires_checkpoint(tmp_path):
    c=client(tmp_path)
    j=c.post('/api/experiments',json={}).json()
    c.post('/api/jobs/'+j['id']+'/cancel')
    r=c.post('/api/jobs/'+j['id']+'/retry',json={'resume':True})
    assert r.status_code==400
    r=c.post('/api/jobs/'+j['id']+'/retry',json={'resume':False})
    assert r.status_code==200 and r.json()['id']!=j['id']


def test_skill_validation_and_dependency(tmp_path):
    c=client(tmp_path)
    assert c.post('/api/benchmarks',json={'skill_ids':['../etc']}).status_code==400
    assert c.post('/api/benchmarks',json={'context_length':4096,'max_new_tokens':4096}).status_code==422
    j=c.post('/api/experiments',json={}).json()
    b=c.post('/api/benchmarks',json={'model_run':j['id']}).json()
    assert b['dependency']==j['id']
