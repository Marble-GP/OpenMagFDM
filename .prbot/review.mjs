#!/usr/bin/env node
// prbot review runner
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

import { loadConfig } from "./lib/config.mjs";
import { filterDiff, truncateText } from "./lib/diff.mjs";
import { buildInstructions, buildReviewBody } from "./lib/prompt.mjs";
import { must, sh } from "./lib/shell.mjs";

export async function runReview(options) {
  const config = loadConfig(options.config ?? ".prbot.json");
  const log = (m) => console.log(`[prbot] ${m}`);

  const pr = options.pr;
  if (!pr || !/^\d+$/.test(String(pr))) throw new Error("--pr <番号> を指定してください");

  const meta = JSON.parse(
    must("gh", ["pr", "view", String(pr), "--repo", config.repo, "--json",
      "number,title,body,author,isDraft,baseRefName,headRefName,additions,deletions,changedFiles"]).stdout,
  );

  if (meta.isDraft && config.skipDrafts) {
    log(`PR #${meta.number} はドラフトのためスキップします`);
    return { skipped: true, reason: "draft" };
  }

  const rawDiff = must("gh", ["pr", "diff", String(pr), "--repo", config.repo]).stdout;
  const filtered = filterDiff(rawDiff, config.excludePaths);
  if (!filtered.text.trim()) {
    log(`PR #${meta.number} にレビュー対象の差分がないためスキップします`);
    return { skipped: true, reason: "no-diff" };
  }
  const { text: diffText, truncated } = truncateText(filtered.text, config.maxDiffChars);

  const workDir = mkdtempSync(join(tmpdir(), "prbot-"));
  try {
    const instructionsPath = join(workDir, "instructions.md");
    const diffPath = join(workDir, "diff.patch");
    writeFileSync(instructionsPath, buildInstructions({
      language: config.language, meta, truncated, excludedFiles: filtered.excluded,
    }), "utf8");
    writeFileSync(diffPath, diffText, "utf8");

    log(`PR #${meta.number} をレビュー中（model: ${config.model}）`);
    const run = sh("opencode", [
      "run", "--standalone", "--auto", "--agent", "plan", "--model", config.model,
      "--file", instructionsPath, "--file", diffPath,
      "Follow the attached instructions and review the attached diff.",
    ], { timeout: config.timeoutMinutes * 60 * 1000 });
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
    must("gh", ["pr", "review", String(pr), "--repo", config.repo, "--comment", "--body-file", bodyPath]);
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
