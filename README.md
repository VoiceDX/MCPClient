# MCPClient

このプロジェクトは、Model Context Protocol (MCP) サーバを活用してユーザの目的達成を支援する ReAct 型のPythonエージェントです。OpenAI の Chat Completion API を用いて計画・実行・評価のループを回します。

## 主な特徴

- `config/system_prompt.txt` にシステムプロンプトを記述し、エージェントの振る舞いを制御
- `mcp_servers.json` で MCP サーバの登録情報を管理
- LLM による実行計画の立案と、MCP サーバを利用した手段の実行
- 実行履歴を保持し、最大 10 回までの ReAct ループで目的の達成を評価

## セットアップ

1. OpenAI API キーを `OPENAI_API_KEY` 環境変数に設定してください。
2. 必要に応じて `config/system_prompt.txt` と `mcp_servers.json` を調整してください。
3. 依存ライブラリをインストールします。

```bash
pip install -r requirements.txt
```

> **Note:** デフォルトでは `openai` パッケージのみを利用します。追加の MCP クライアント実装が必要な場合は別途導入してください。

## 使い方

```bash
python main.py "ブラウザで特定の記事を検索して内容を要約したい"
```

`goal` を省略すると、実行時にプロンプト入力が求められます。

ログレベルやモデルはオプションで調整できます。

```bash
python main.py --model gpt-4.1 --log-level DEBUG
```

## 実行フロー

1. LLM が目的と履歴に基づいて MCP サーバを利用する計画を作成
2. 計画の各ステップを実行し、結果を履歴に追加
3. 最新の結果をもとに LLM が目的達成を評価
4. 達成していなければ履歴を含む新しいプロンプトで再計画
5. 最大 10 回繰り返し、成功または警告で終了

> 現在の `MCPClient` 実装では実サーバへの接続部分はモック化されています。実際の MCP サーバと連携する場合は `agent/mcp_client.py` の `execute` メソッドを拡張してください。

## 設定ファイル

- `config/system_prompt.txt` — システムプロンプト
- `mcp_servers.json` — MCP サーバ登録。Claude Desktop 互換の JSON 形式です。

## 開発メモ

- コードは `agent/` ディレクトリで機能単位に分割されています。
- `main.py` が CLI エントリポイントです。
- 単体テスト等を追加する場合は `tests/` ディレクトリを作成してください。
