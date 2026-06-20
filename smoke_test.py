from pipeline.utils import load_config, ensure_dirs, get_env
from dotenv import load_dotenv
load_dotenv()

cfg = load_config()
ensure_dirs(cfg)
print("Config loaded OK")
print("Sources:")
for lang, lcfg in cfg["sources"].items():
    print(f"  {lang}: {len(lcfg['urls'])} videos ({lcfg['language_code']})")

print()
print("Quality thresholds:")
for k, v in cfg["quality"].items():
    print(f"  {k}: {v}")

print()
try:
    key = get_env("SARVAM_API_KEY")
    hf = get_env("HF_TOKEN")
    print(f"SARVAM_API_KEY loaded: ...{key[-6:]}")
    print(f"HF_TOKEN loaded:       ...{hf[-6:]}")
except Exception as e:
    print(f"Env error: {e}")

print()
print("Directories created:")
for name, path in cfg["paths"].items():
    import pathlib
    p = pathlib.Path(path)
    print(f"  {name}: {p} ({'OK' if p.exists() else 'MISSING'})")
