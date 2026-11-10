"""Opt-in: actual LFM3 architecture, tiny random weights; no full model download."""
import os
import pytest

@pytest.mark.skipif(not os.getenv('SPL_FT_MODEL_CODE_PATH'),reason='Opt-in native LFM3 CPU training smoke')
@pytest.mark.parametrize('method',['full','lora'])
def test_native_architecture_trainer_checkpoint_and_reload(tmp_path,method):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM,Trainer,TrainingArguments
    from peft import get_peft_model,LoraConfig
    c=AutoConfig.from_pretrained(os.environ['SPL_FT_MODEL_CODE_PATH'],trust_remote_code=True)
    c.hidden_size=32;c.intermediate_size=64;c.head_dim=8;c.num_hidden_layers=3
    c.layer_types=['conv','sliding_attention','full_attention']
    c.num_attention_heads=4;c.num_key_value_heads=2;c.vocab_size=256
    c.bos_token_id=1;c.eos_token_id=2;c.pad_token_id=0;c.sliding_window=16;c.use_cache=False
    m=AutoModelForCausalLM.from_config(c,trust_remote_code=True,attn_implementation='sdpa')
    if method=='lora':
        m=get_peft_model(m,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj','o_proj','in_proj','out_proj','w1','w2','w3'],task_type='CAUSAL_LM'))
        m.enable_input_require_grads()
    x=torch.randint(3,256,(4,32));labels=x.clone();labels[:,:10]=-100
    data=[{'input_ids':a,'labels':b} for a,b in zip(x,labels)]
    args=TrainingArguments(output_dir=str(tmp_path),max_steps=2,use_cpu=True,per_device_train_batch_size=1,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        report_to='none',save_steps=1,save_total_limit=2,logging_steps=1,remove_unused_columns=False,
        prediction_loss_only=True,include_num_input_tokens_seen=True,optim='adamw_torch')
    trainer=Trainer(model=m,args=args,train_dataset=data)
    result=trainer.train()
    assert result.global_step==2
    assert (tmp_path/'checkpoint-2'/'trainer_state.json').exists()
    assert (tmp_path/'checkpoint-2'/'optimizer.pt').exists()
    assert torch.isfinite(torch.tensor(result.training_loss))
    checkpoint=tmp_path/'checkpoint-2'
    if method=='lora':
        assert (checkpoint/'adapter_config.json').exists()
        from peft import PeftModel
        loaded=PeftModel.from_pretrained(AutoModelForCausalLM.from_config(c,trust_remote_code=True,attn_implementation='sdpa'),checkpoint)
    else:
        assert (checkpoint/'config.json').exists()
        loaded=AutoModelForCausalLM.from_pretrained(checkpoint,trust_remote_code=True)
    with torch.no_grad():
        assert torch.isfinite(loaded(input_ids=x[:1],labels=labels[:1],use_cache=False).loss)
    trainer.args.max_steps=3
    resumed=trainer.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.global_step==3
