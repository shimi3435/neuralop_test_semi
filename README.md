# neuralop_test_semi

GitHubとpython仮想環境とneuraloperatorライブラリを使用する

1. neruralopを動かしてみる

2. （GNOTの論文コードを動かしてみる）

3. NVIDIA physicsNeMoを動かしてみる

4. （PS上で動かしてみる）

## neuralopライブラリ編

### python仮想環境構築

pyenvのインストール https://github.com/pyenv/pyenv

インストール後

```bash
pyenv install 3.10.4 #使いたいpython(今回の場合3.10.4)のインストール
pyenv local 3.10.4 #このディレクトリで使用するpyenv/pythonのバージョンをpython3.10.4に設定
python -m venv .venv #.venvディレクトリに仮想環境を作成

. .venv/bin/activate #仮想環境に入る
```

仮想環境から出るコマンド
```bash
deactivate
```

### neuralopライブラリのインストール
GitHub: https://github.com/neuraloperator/neuraloperator

Installationの上側推奨
```bash
git clone https://github.com/NeuralOperator/neuraloperator
cd neuraloperator
pip install -e .
pip install -r requirements.txt
```

neuralopのドキュメント https://neuraloperator.github.io/dev/index.html

exampleに.py形式と.ipynb形式（ジュピターノートブック形式）のふたつがある

### 自分のデータを使ってFNO学習したい場合

exampleではload_darcy_flow_small()を用いてデータを読み込んでいる

neuralop/data/datasets/darcy.pyに書かれているload_darcy_flow_small()を見に行く

load_darcy_flow_small()を見ると、DarcyDatasetクラスが定義されており、そこでPTDatasetクラスが使用されている

同じディレクトリにあるpt_dataset.pyに書かれているPTDatasetクラスを見ると、# Load train data部分と# load test data部分が見つかる

この部分でtorch.loadされているpytoch tensorを自分のものに置き換えると自分のデータを使用できる（はず）

## GNOTの論文コード編

論文：https://arxiv.org/abs/2302.14376

### python仮想環境構築

pythonのバージョンが3.10以上だとエラーが出たので注意。python3.9.21で実行を確認。

```bash
pyenv install 3.9.21
pyenv local 3.9.21
python -m venv .venv_GNOT

. .venv_GNOT/bin/activate #仮想環境に入る
```

### WSL上でGPUを使用する（TensorFlow）（以前より複雑ではない？）

参考文献

https://docs.nvidia.com/cuda/wsl-user-guide/index.html#gpu-accelerated-computing

https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin

1. Windowsで普通のNVIDIAドライバーをダウンロード
https://www.nvidia.com/en-us/drivers/

2. CUDA Toolkitをダウンロード（今回は12.4 最後の所をcuda-toolkit-12-4にすればOK）（バージョン変更は https://www.nemotos.net/?p=2374 参照）
https://developer.nvidia.com/cuda-downloads

3. CUDAのPATHを通す（~/.bashrcに下記を追記）
```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

4. cuDNNのダウンロード（もしかしたら不必要）（今回は9.7.1）
https://developer.nvidia.com/cudnn-downloads

5. TensorFlowのインストール
```bash
pip install tensorflow-cpu==2.10
pip install tensorflow-directml-plugin
```

### モジュールのインストール

```bash
git clone git@github.com:HaoZhongkai/GNOT.git
cd GNOT
pip install -r requirements.txt
```

requirements.txtだけでは足りないので以下をインストール（condaならいらないかも？）
```bash
pip install torchdata==0.9.0 #最新版だとダメそう
pip install pandas==2.0.0 #requirements.txtのnumpyで使える範囲
pip install pyyaml
pip install pydantic
pip uninstall torch
pip install torch==2.4.0 
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html #cuda12.4とpytorch2.4.xに対応したdgl
```

### プログラムの実行

GNOTのGitHub https://github.com/HaoZhongkai/GNOT/tree/master を参考に、dataディレクトリとその中にlogsとcheckpointsディレクトリを作成し、
Dataset Linkからお好きなデータセットをダウンロードし、dataディレクトリに移動させる（今回はheat2dの場合）

```
python train.py --gpu 0 --dataset heat2d --use-normalizer unit  --normalize_x unit --component all --comment rel2  --loss-name rel2 --epochs 500 --batch-size 4 --model-name CGPT --optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle  --grad-clip 1000.0   --n-hidden 128 --n-layers 3  --use-tb 0 
```

## NVIDIA PhysicsNeMo編

TOPページ https://developer.nvidia.com/physicsnemo

Getting Started https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html

サーバーで動かす場合，個人のDockerの環境構築とかは必要ない（はず）

dockerの使い方はググればいろいろな人が記事をかいてるのでここでは省略

Google Colabでとりあえず動かせそう（ランタイムのタイプをCPUからT4 GPUに変更する）

PhysicsNemo/ColabにあるのがGoogle Colab用のノートブック

Google Colabで動かしやすいようにするかも？（参照：https://qiita.com/haraso_1130/items/20a50b0474c88781dcc1）

```
PhysicsNemo/Colab/
    test_PhysicsNeMo.ipynb      #pipインストールしたphysicsnemoとsymの動作確認
    tutorial1_FNO.ipynb         #physicsnemoのチュートリアル
    Intro_PhysicsNeMoSym.ipynb  #physicsnemo-symのイントロ
    main.ipynb                  #Colabで使いやすいようにするための第一歩
```

下記はGPU付きの個人所有WindowsPC向けの環境構築

### Dockerのインストール

https://docs.docker.com/engine/install/ubuntu/

上記URLのInstall using the apt repositoryでやるのがよさそう

### NVIDIA Container Toolkitのインストール

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Prerequisitesの2番について，WSLでCUDAをインストールしていない場合，GNOTの論文コード編/WSLでGPUを使用する の3番（もしくは4番）までやればいいはず

### NIVIA PhysicNeMo Containerのイメージのダウンロード

DockerとNVIDIA Container Toolkitがインストールできれば，以下のコマンドでPhysicNeMo Containerのイメージをダウンロードできる

```bash
sudo docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.03    #25.03は任意のバージョンにする
```

sudoを使わない場合は，dockerグループを作ってユーザーをdockerグループに追加する https://docs.docker.com/engine/install/linux-postinstall/

### NIVIA PhysicNeMo Containerを生成し，起動を行う

```bash
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --rm nvcr.io/nvidia/physicsnemo/physicsnemo:25.03 bash    #--runtime nvidia ではエラーが出るので --gpus all に変更している
```

うまくいけば以下のような出力がでるはず

```bash
========================
== NVIDIA PhysicsNeMo ==
========================

NVIDIA Release 25.03 (build 25392890)
Modulus PyPi Version 1.0.0 (Git Commit: f87c6be)
Modulus Sym PyPi Version 2.0.0 (Git Commit: 8f6ad0b)
Container image Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2014-2024 Facebook Inc.
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
Copyright (c) 2015      Google Inc.
Copyright (c) 2015      Yangqing Jia
Copyright (c) 2013-2016 The Caffe contributors
All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

root@3df62933a5f5:/workspace#
```

## プラズマシミュレータのGPUを使用した実行について

PSのポータルサイトに入るために必要なものは，はがきに書いてあるIDとPWとスマホアプリのDuo Mobile

その後ポータルサイトから公開鍵の登録をすればPSにsshできる

PSにsshするのに必要なのは，登録した公開鍵に対応した秘密鍵とワンタイムパスワード（スマホのGoogle Authenticatorにした）

ユーザーガイドによると，ホームディレクトリ以下はログイン時に必要なファイル(.sshの中身とか)を置いて，それ以外のファイルはdataディレクトリにおいてほしいらしい

t-shimizuは/data/t-shimizu/workspace/python3/neuralop_test_semi/みたいな感じにした

pythonの仮想環境を使用したいが，pyenvは使えなさそう（OSに必要なライブラリがはいってなさそう）

システムにpython3.9とpython3.12が用意されているため，3.9と3.12の仮想環境は作れる

```bash
cd /data/t-shimizu/workspace/python3/neuralop_test_semi/

python3.9 -m venv .venv_3_9
. .venv_3_9/bin/activate
deactivate

python3.12 -m venv .venv_3_12
. .venv_3_12/bin/activate
deactivate
```

PSにsshして最初にいるところはフロントシステム部であり，GPUを使用するためには，フロントシステム部からGPUが搭載されている計算サーバへジョブを投入してジョブの完了を待つ必要がある

その際に使用するのがジョブスクリプトと呼ばれるもの（PS_testのPS_jobscript_helloworld.shなど）

フロントシステム部からジョブを投げるコマンドがqsub，投入したジョブの状態を見るコマンドがqstat

```bash
qsub PS_jobscript_helloworld.sh
qstat -x #-xは過去に投入したジョブの状態も見れる 実行中のものを見るだけならqstat単体でOK
```

ジョブスクリプトの記述によるが，基本的にはジョブが終了すると標準出力結果が[タスクの名前].o[ジョブ番号]，標準エラー出力結果が[タスクの名前].e[ジョブ番号]として返ってくる．

pytorchをベースにしているneuraloperatorライブラリのFNO学習において，GPUを使用した実行は確認できた（https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-pytorch.html 参照）

physicsnemoはダメそう；；

## Tips

### おすすめのコードエディタ

（WindowsならWSL + ）VSCode 

https://azure.microsoft.com/ja-jp/products/visual-studio-code

### wandb （Weights & Biases）

きれいに可視化できるやつ

https://wandb.ai/site/ja/

1. wandbでアカウント登録（無料）
2. 設定からAPIキーの取得 https://wandb.ai/settings
3. pip install wandb
4. wandb login
5. APIキーをペースト

