#!/usr/bin/env node
// prbot review runner
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

import { loadConfig } from "./lib/config.mjs";
import { filterDiff, truncateText } from "./lib/diff.mjs";
import { buildInstructions, buildReviewBody } from "./lib/prompt.mjs";
import { must, sh } from "./lib/shell.mjs";

// OpenCode 子プロセスへ渡してよい環境変数の許可リスト。
// GitHub のトークン（GH_TOKEN / GITHUB_TOKEN）やその他の GITHUB_* は決して渡さない。
const ENV_ALLOWLIST = [
  // 実行環境
  "PATH", "Path", "PATHEXT", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
  "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR",
  "SystemRoot", "SYSTEMROOT", "windir", "ComSpec", "COMSPEC",
  "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "NO_COLOR",
  // OpenCode の設定/認証情報の探索先
  "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
  "OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR",
];

// プロバイダの API キーのみを許可する（GH_TOKEN 等は *_API_KEY ではないため除外される）
function isProviderKey(name) {
  return /_API_KEY$/.test(name) && !/^(GH|GITHUB)_/.test(name);
}

export function buildOpencodeEnv(model, source = process.env) {
  const env = {};
  for (const key of ENV_ALLOWLIST) {
    if (source[key] !== undefined) env[key] = source[key];
  }
  for (const [key, value] of Object.entries(source)) {
    if (isProviderKey(key) && value !== undefined) env[key] = value;
  }
  // 念のため: GitHub のトークンは絶対に含めない
  for (const key of Object.keys(env)) {
    if (/^(GH|GITHUB)_/.test(key) || key === "GH_TOKEN" || key === "GITHUB_TOKEN") {
      delete env[key];
    }
  }
  env.OPENCODE_MODEL = model;
  return env;
}

export async function runReview(options) {
  const config = loadConfig(options.config ?? ".prbot.json");
  const log = (m) => console.log(`[prbot] ${m}`);

  const pr = options.pr;
  if (!pr || !/^\d+$/.test(String(pr))) throw new Error("--pr <番号> を指定してください");

  // gh コマンドには GH_TOKEN を含む完全な環境を渡す（信頼済みツール）
  const ghEnv = process.env;

  const meta = JSON.parse(
    must("gh", ["pr", "view", String(pr), "--repo", config.repo, "--json",
      "number,title,body,author,isDraft,baseRefName,headRefName,additions,deletions,changedFiles"], { env: ghEnv }).stdout,
  );

  if (meta.isDraft && config.skipDrafts) {
    log(`PR #${meta.number} はドラフトのためスキップします`);
    return { skipped: true, reason: "draft" };
  }

  const rawDiff = must("gh", ["pr", "diff", String(pr), "--repo", config.repo], { env: ghEnv }).stdout;
  const filtered = filterDiff(rawDiff, config.excludePaths);
  if (!filtered.text.trim()) {
    log(`PR #${meta.number} にレビュー対象の差分がないためスキップします`);
    return { skipped: true, reason: "no-diff" };
  }
  const { text: diffText, truncated } = truncateText(filtered.text, config.maxDiffChars);

  // 作業ファイルはリポジトリ内の .prbot/work/ に置く。
  // システムの一時ディレクトリを使うとエージェントの外部ディレクトリ参照が必要になるため、
  // プロジェクト内に閉じることで external_directory を全面拒否できる。
  const workRoot = join(process.cwd(), ".prbot", "work");
  mkdirSync(workRoot, { recursive: true });
  const workDir = mkdtempSync(join(workRoot, "run-"));
  try {
    const instructionsPath = join(workDir, "instructions.md");
    const diffPath = join(workDir, "diff.patch");
    writeFileSync(instructionsPath, buildInstructions({
      language: config.language, meta, truncated, excludedFiles: filtered.excluded,
    }), "utf8");
    writeFileSync(diffPath, diffText, "utf8");

    log(`PR #${meta.number} をレビュー中（model: ${config.model}）`);
    // prbot 専用エージェント（.opencode/agents/prbot-review.md）を使用する。
    // 権限はこのエージェントにのみ適用され、開発セッションには影響しない。
    const agentFile = join(process.cwd(), ".opencode", "agents", "prbot-review.md");
    if (!existsSync(agentFile)) {
      throw new Error(
        ".opencode/agents/prbot-review.md が見つかりません。\n" +
        "`prbot init` を実行して生成し、コミットしてください。");
    }
    // OpenCode には最小限の環境変数のみを渡す。
    // GH_TOKEN / GITHUB_TOKEN は含めないため、diff 経由のプロンプトインジェクションで
    // トークンを窃取・悪用されることはない。
    const run = sh("opencode", [
      "run", "--standalone", "--auto", "--agent", "prbot-review", "--model", config.model,
      "--file", instructionsPath, "--file", diffPath,
      "Follow the attached instructions and review the attached diff.",
    ], { timeout: config.timeoutMinutes * 60 * 1000, env: buildOpencodeEnv(config.model) });
    if (run.status !== 0) {
      throw new Error(`opencode run が失敗しました (exit ${run.status}): ${run.stderr.trim().slice(-2000)}`);
    }
    if (!run.stdout.trim()) throw new Error("OpenCode の出力が空でした");

    const body = buildReviewBody({
      model: config.model, review: run.stdout, truncated, excludedFiles: filtered.excluded, language: config.language,
    });

    if (options.dryRun) {
      process.stdout.write(body + "\n");
      log("dry-run のため投稿していません");
      return { posted: false, body };
    }

    const bodyPath = join(workDir, "review.md");
    writeFileSync(bodyPath, body, "utf8");
    must("gh", ["pr", "review", String(pr), "--repo", config.repo, "--comment", "--body-file", bodyPath], { env: ghEnv });
    log(`PR #${meta.number} にレビューを投稿しました`);
    return { posted: true };
  } finally {
    rmSync(workDir, { recursive: true, force: true });
  }
}

export async function main(argv) {
  const { values } = parseArgs({
    args: argv,
    options: {
      pr: { type: "string" },
      config: { type: "string", default: ".prbot.json" },
      "dry-run": { type: "boolean", default: false },
    },
  });
  await runReview({ pr: values.pr, config: values.config, dryRun: values["dry-run"] });
}

const isEntry = process.argv[1] && /[\\/]review\.mjs$/.test(process.argv[1]);
if (isEntry) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(`[prbot] エラー: ${error.message}`);
    process.exitCode = 1;
  });
}
