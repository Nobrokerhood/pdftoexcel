# Google Resource Bootstrap

Check Google resources without modifying anything:

```powershell
python tools/bootstrap_google_resources.py
```

Equivalent explicit command:

```powershell
python tools/bootstrap_google_resources.py --check
```

Create missing spreadsheet tabs and header rows only:

```powershell
python tools/bootstrap_google_resources.py --bootstrap
```

Create the Drive root folder during bootstrap if one is not configured:

```powershell
python tools/bootstrap_google_resources.py --bootstrap --create-folders
```

Mock mode avoids live Google calls and is useful for automated checks:

```powershell
python tools/bootstrap_google_resources.py --check --mock
```

The tool reports only safe status values:

- `PASS`
- `READY`
- `MISSING`
- `MISMATCH`
- `NO_ACCESS`

It does not print service-account JSON, Gemini keys, OAuth tokens, or spreadsheet/folder IDs.

Bootstrap is additive. It does not delete tabs, overwrite existing rows, or replace mismatched headers.
