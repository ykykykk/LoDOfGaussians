import functools
import os


def prepare_gsplat_windows():
    if os.name != "nt":
        return

    import torch.utils.cpp_extension as cpp_extension

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
