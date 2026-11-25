import { basename } from "node:path";

export function globToRegExp(glob) {
  let re = "";
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i];
    if (c === "*") {
      if (glob[i + 1] === "*") {
        re += ".*";
        i++;
      } else {
        re += "[^/]*";
      }
    } else if (c === "?") {
      re += "[^/]";
    } else {
      re += c.replace(/[.+^${}()|[\]\\]/g, "\\$&");
    }
  }
  return new RegExp(`^${re}$`);
}

export function isExcluded(path, patterns) {
  return patterns.some((pattern) => {
    const target = pattern.includes("/") ? path : basename(path);
    return globToRegExp(pattern).test(target);
  });
}

export function splitDiff(diff) {
  return diff
    .split(/^(?=diff --git )/m)
    .filter(Boolean)
    .map((text) => {
      const match = text.match(/^diff --git a\/(.+?) b\/(.+)$/m);
      let path = match ? match[2].trim() : "";
      // git quotes paths containing special characters
      if (path.startsWith('"') && path.endsWith('"')) {
        path = path.slice(1, -1).replace(/\\(.)/g, "$1");
      }
      return { path, text };
    });
}

export function filterDiff(diff, excludePaths) {
  const files = splitDiff(diff);
  const kept = files.filter((file) => !isExcluded(file.path, excludePaths));
  return {
    text: kept.map((file) => file.text).join(""),
    total: files.length,
    excluded: files.length - kept.length,
  };
}

export function truncateText(text, maxChars) {
  if (text.length <= maxChars) return { text, truncated: false };
  return { text: `${text.slice(0, maxChars)}\n\n... (差分は上限のため切り捨てました)\n`, truncated: true };
}
