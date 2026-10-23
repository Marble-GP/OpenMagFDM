# MCP integration

This manual covers the local integration shipped with v1.7.2. Agents access a
fixed workspace and perform explicitly permitted operations. See [REST API](REST_API.md)
for ordinary HTTP clients.

## Setup

1. Start `webui-server.exe`, `./webui-server`, or the source server with `npm start`.
2. Open **Help & Integrations** and check **Assigned workspace**.
3. Pair using the terminal's **MCP management pairing code**. Explicitly choose **Trust this browser for 30 days** if needed.
4. Name a connection and **Create connection**, initially with read permission only.
5. Apply **Enable MCP**; verify the listening state and URL. MCP initially defaults to OFF.
6. Run **Test**, then register the values under **Settings** in your MCP client.

Retrieve a lost pairing code by running the same installation with `--mcp-admin-code`.
Different installation paths or OS accounts do not share profiles. Default MCP
URL: `http://127.0.0.1:3001/mcp`. Change the port in its management UI; an MCP
bind conflict does not stop the WebUI.

## Transports

| Transport | Client setup |
|---|---|
| Streamable HTTP | Displayed URL and `Authorization: Bearer <token shown at creation>` |
| stdio | Displayed `stdio.command` and `stdio.args` in the client's configuration format |

Packaged stdio uses `webui-server[.exe] --mcp-stdio --connection <id>`.
Source stdio uses Node, the absolute launcher path and remaining arguments.
Copy the displayed settings. Stdio bridges to an already running server; it
does not start the server or enable MCP. A raw HTTP token is shown only at
creation; a connection ID is not a token. Restart client connections after changes.

## Workspaces and permissions

A connection is fixed to its assigned workspace. Grant only the required
`read`, `edit`, `images`, `run`, `cancel`, `delete` and `replace` permissions.
File mutations use retrieved revisions; submission uses an idempotency key.
Shared-mode ownership and enabled-account checks use the common server policy.

An initial request can be:

```text
Use workspace_status and list the assigned workspace's inputs and results.
Summarize one result's conditions, convergence and warnings. Do not edit files
or start analyses yet.
```

Disconnecting, revoking or turning MCP OFF does not cancel running analyses.
Server shutdown terminates solvers. The server stores its queue/state in local
private JSON; workspace/result discovery reads actual directories. See
[Operations](OPERATIONS.md#data-and-metadata).

## Shared HTTPS limitation

MCP remains local. Its management UI uses a loopback HTTP Origin policy and is
not fully integrated with shared HTTPS administration. Create profiles and
save client settings in local mode before switching. Keep the same OS account
and installation path, and assign workspaces to enabled accounts. Web login/CSRF
and MCP management pairing/CSRF are separate mechanisms.

## Troubleshooting

| Symptom | Check |
|---|---|
| GUI Test passes, client fails | Client URL/token or command/arguments; restart connection |
| Connection refused | Server running, MCP enabled/listening, port conflict |
| Stdio failure | Same OS account/install, absolute executable path and arguments |
| Missing files | Assigned workspace, ownership, permissions and browser workspace ID |
| 409 conflict | Retrieve revision and rebuild the operation |
| Lost token | Revoke the old connection and create a replacement |

Release CI tests packaged HTTP/stdio on three OSes. Long-duration memory,
all manual browser interactions and first-time agent usability require separate
acceptance. Historical records: [integration archive](archive/integrations/MCP_INTEGRATION_BEFORE_v1.7.2.md)
and [external-client acceptance guide](archive/acceptance/MCP_ACCEPTANCE_TEST.md).
