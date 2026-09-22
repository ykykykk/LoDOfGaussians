import functools
import os
import sys


def prepare_gsplat_windows():
    if os.name != "nt":
        return

    scripts_dir = os.path.dirname(sys.executable)
    if scripts_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ["PATH"]

    import torch.utils.cpp_extension as cpp_extension

    os.environ.setdefault("VSLANG", "1033")
    # MSVC may emit an OEM/ANSI banner even when Python uses UTF-8. Version
    # parsing is ASCII; preserve diagnostics without failing the build on bytes.
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    original = cpp_extension._jit_compile
    if getattr(original, "_alod_windows_compatible", False):
        return

    def sanitize(flags):
        if flags is None:
            return None
        return ["/O2" if flag == "-O3" else flag for flag in flags if flag != "-Wno-attributes"]

    @functools.wraps(original)
    def compatible_jit_compile(*args, **kwargs):
        args = list(args)
        if len(args) > 2:
            args[2] = sanitize(args[2])
        elif "extra_cflags" in kwargs:
            kwargs["extra_cflags"] = sanitize(kwargs["extra_cflags"])
        return original(*args, **kwargs)

    compatible_jit_compile._alod_windows_compatible = True
    cpp_extension._jit_compile = compatible_jit_compile
