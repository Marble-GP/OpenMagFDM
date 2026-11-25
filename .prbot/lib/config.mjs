import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

export const REPO_PATTERN = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
export const MODEL_PATTERN = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_./:-]+(#\w+)?$/;

export function defaultConfig() {
  return {
    language: "ja",
    skipDrafts: true,
    maxDiffChars: 120000,
    excludePaths: [],
    timeoutMinutes: 15,
  };
}

export function validateConfig(config) {
  if (typeof config.repo !== "string" || !REPO_PATTERN.test(config.repo)) {
    throw new Error("設定の repo は owner/name 形式で指定してください");
  }
  if (typeof config.model !== "string" || !MODEL_PATTERN.test(config.model)) {
    throw new Error("設定の model は provider/model 形式で指定してください");
  }
  const merged = { ...defaultConfig(), ...config };
  if (!Array.isArray(merged.excludePaths) || !merged.excludePaths.every((p) => typeof p === "string")) {
    throw new Error("設定の excludePaths は文字列の配列で指定してください");
  }
  if (!Number.isInteger(merged.maxDiffChars) || merged.maxDiffChars < 1000) {
    throw new Error("設定の maxDiffChars は 1000 以上の整数で指定してください");
  }
  if (!Number.isInteger(merged.timeoutMinutes) || merged.timeoutMinutes < 1) {
    throw new Error("設定の timeoutMinutes は 1 以上の整数で指定してください");
  }
  if (merged.secretName && !/^[A-Z_][A-Z0-9_]*$/.test(merged.secretName)) {
    throw new Error("設定の secretName は大文字英数字とアンダースコアで指定してください");
  }
  return merged;
}

export function loadConfig(path = ".prbot.json") {
  const full = resolve(path);
  if (!existsSync(full)) throw new Error(`設定ファイルが見つかりません: ${full}\n先に \`prbot init\` を実行してください。`);
  let raw;
  try {
    raw = JSON.parse(readFileSync(full, "utf8"));
  } catch (e) {
    throw new Error(`設定ファイルの JSON が不正です: ${full}\n${e.message}`);
  }
  return validateConfig(raw);
}

export function saveConfig(config, path = ".prbot.json") {
  writeFileSync(resolve(path), JSON.stringify(config, null, 2) + "\n", "utf8");
}
