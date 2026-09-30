# training reward func
grpo_v2的相关数据集(train & eval)reward func应该都是math reward

coverage, difficulty_sae, difficulty_baseline：reward func改成:
    deepscaler对应deepscaler，
    math+aime+amc都是math reward，
    minerva+olympiad都是prime math

在qwen2.5上重新training，reward func改成：
    deepscaler+minerva+olympiad对应deepscaler，
    math+aime+amc都是math reward
    !!!when reward, change reward func
    这个模型的max model len是4096(maybe)，所以配置更改了

# eval
v1和v2_old都是在gainrl的vllm配置下进行的推理，也就是max len在3000，和一些其他变动
v2是新改的vllm配置下进行的推理，max len=10240

old means: max len更小


可以直接在训练时进行val，具体见sh文件，list传入多个val files，但是记得生成符合格式的parquet文件

2.7 15:30之后，新的eval代码（gemini）

