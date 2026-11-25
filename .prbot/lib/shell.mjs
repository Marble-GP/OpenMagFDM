// シェル実行ユーティリティ（外部依存なし）
import { spawnSync } from "node:child_process";

// Node の shell:true に関する deprecation 警告を抑制（引数は検証済みのため）
process.noDeprecation = true;

export function sh(cmd, args, options = {}) {
  const useShell = process.platform === "win32";
  // shell 経由の場合は空白を含む引数を引用符で保護
  const finalArgs = useShell
    ? args.map((a) => (typeof a === "string" && /\s/.test(a) ? `"${a.replace(/"/g, '\\"')}"` : a))
    : args;
  const result = spawnSync(cmd, finalArgs, {
    encoding: "utf8",
    input: options.input,
    timeout: options.timeout,
    maxBuffer: options.maxBuffer ?? 256 * 1024 * 1024,
    shell: useShell,
    cwd: options.cwd,
    // env が指定されない場合は process.env を継承（後方互換）
    env: options.env ?? process.env,
  });
  return {
    status: result.status,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    error: result.error,
  };
}

export function must(cmd, args, options) {
  const result = sh(cmd, args, options);
  if (result.error) throw new Error(`${cmd} の実行に失敗しました: ${result.error.message}`);
  if (result.status !== 0) {
    const detail = result.status === null
      ? `シグナルにより終了しました: ${result.stderr.trim().slice(-2000)}`
      : `exit ${result.status}: ${result.stderr.trim().slice(-2000)}`;
    throw new Error(`${cmd} が失敗しました (${detail})`);
  }
  return result;
}

export function hasCommand(cmd) {
  const r = sh(process.platform === "win32" ? "where" : "which", [cmd]);
  return r.status === 0;
}
