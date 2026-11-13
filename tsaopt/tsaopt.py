#!/usr/bin/env python
import sys
import subprocess
import os

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python tsaopt.py [command] [options...]")
        print("\nCommands:")
        print("  simulate           Run base transient stability simulation (run_ts.py)")
        print("  optimize-direct    Run direct collocation or iLQR optimization (run_tsa_optimization.py)")
        print("  optimize-modular   Run modular system step optimization (test_modular_opt.py)")
        print("\nUse 'python tsaopt.py [command] --help' for command-specific options.")
        sys.exit(1 if len(sys.argv) < 2 else 0)

    command = sys.argv[1]
    args = sys.argv[2:]

    config_args = []
    if '--config' in args:
        idx = args.index('--config')
        if idx + 1 < len(args):
            config_path = args[idx + 1]
            args.pop(idx)
            args.pop(idx)
            
            import json
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                    
                for k, v in cfg.items():
                    if k == 'plots':
                        if '--plot-list' not in args:
                            config_args.extend(['--plot-list', config_path])
                        if 'tag' in v and '--plot-tag' not in args:
                            config_args.extend(['--plot-tag', str(v['tag'])])
                        if 'main_title' in v and '--plot-title' not in args:
                            config_args.extend(['--plot-title', str(v['main_title'])])
                        if v.get('enabled') is False and '--no-plots' not in args:
                            config_args.append('--no-plots')
                        continue

                    flag = f"--{k.replace('_', '-')}"
                    if flag in args: 
                        continue
                        
                    if isinstance(v, bool):
                        if v:
                            config_args.append(flag)
                    else:
                        config_args.extend([flag, str(v)])
            except Exception as e:
                print(f"Failed to load or parse config file: {e}")
                sys.exit(1)
                
    args = config_args + args

    cmd_map = {
        'simulate': 'scripts/entrypoints/run_ts.py',
        'optimize-direct': 'scripts/entrypoints/run_tsa_optimization.py',
        'optimize-modular': 'scripts/entrypoints/test_modular_opt.py',
    }

    if command not in cmd_map:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(cmd_map.keys())}")
        sys.exit(1)

    script_path = cmd_map[command]
    
    # Ensure we run from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    try:
        # Run in subprocess to preserve all arg parsing logic and paths cleanly
        result = subprocess.run(
            [sys.executable, script_path] + args,
            cwd=project_root,
            env=env
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(130)

if __name__ == "__main__":
    main()
