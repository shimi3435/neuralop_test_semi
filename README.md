# neuralop_test_semi

GitHubとpython仮想環境とneuraloperatorライブラリを使用する

neruralopとGNOTの論文コードを動かしてみる

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

