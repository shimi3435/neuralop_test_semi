# neuralop_test_semi

GitHubとpython仮想環境とneuralopライブラリを使用する。

## python仮想環境構築

pyenvのインストール（https://github.com/pyenv/pyenv 参照）

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

## neuralopライブラリのインストール
GitHub: https://github.com/neuraloperator/neuraloperator

Installationの上側推奨
```bash
git clone https://github.com/NeuralOperator/neuraloperator
cd neuraloperator
pip install -e .
pip install -r requirements.txt
```

neuralopのドキュメント https://neuraloperator.github.io/dev/index.html

exampleに.py形式と.ipynb形式（ジュピターノートブック形式）のふたつがある。
