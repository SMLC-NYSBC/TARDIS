#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  Robert Kiewisz                                                     #
#  MIT License 2021 - 2026                                            #
#  (CLI wrapper; calls pandorica.stitch under Prosperity 3.0.0)       #
#######################################################################

"""
``tardis_stitch`` — a thin click CLI over :func:`pandorica.stitch.cli.run_stitch`.

The flag set is **derived from pandorica at import time** by walking
``inspect.signature(run_stitch)``: every annotated parameter becomes a flag
whose type, default, and (if available) help text come straight from
pandorica. The wrapper itself does not enumerate pandorica's parameters, so
upgrading pandorica is enough to expose any new flags — no tardis_em release
required.

Caveats:

* ``pandorica`` and ``docstring_parser`` are pulled in by the ``stitch``
  extra (``pip install "tardis_em[stitch]"``). Without them, the CLI still
  loads but explains how to install the stitcher.
* Parameters with no annotation, or whose default is a callable (e.g.
  ``log=print``), are skipped — they're treated as library-only knobs.
* Renaming a pandorica parameter renames the CLI flag too; that's a
  user-visible API change that pandorica should call out in its CHANGELOG.
"""

import inspect
import typing
import warnings

import click

from tardis_em._version import version
from tardis_em.utils.logo import print_error

# Pandorica (Prosperity-licensed, separate maintenance) provides the stitcher
# engine. docstring_parser turns its ``:param:`` lines into per-flag --help
# text. Both come from the ``[stitch]`` extra; without them the CLI still
# loads but only prints an install hint.
try:
    from pandorica.stitch.cli import run_stitch  # type: ignore[import-not-found]
except ImportError:  # noqa: F401
    run_stitch = None

try:
    from docstring_parser import parse as _parse_docstring  # type: ignore[import-not-found]
except ImportError:  # noqa: F401
    _parse_docstring = None

warnings.simplefilter("ignore", UserWarning)


# ---------------------------------------------------------------------------
# Signature → click.Option translation
# ---------------------------------------------------------------------------


def _unwrap_optional(ann):
    """``Optional[T]`` (== ``Union[T, None]``) → ``T``; pass others through.

    Click can't represent ``Optional[int]`` natively — it just needs the
    inner ``int`` and a ``default=None`` to express the same semantics. For
    anything that isn't a 1-arg ``Union`` with ``None``, leave it alone and
    let the option fall back to a string-typed flag.
    """
    if typing.get_origin(ann) is typing.Union:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return ann


def _is_exposable(param: inspect.Parameter) -> bool:
    """Skip params that don't map cleanly to a CLI flag.

    * No annotation → we don't know how to validate input.
    * Callable default (e.g. ``log=print``) → almost certainly a library
      injection point, not a user-facing knob.
    """
    if param.annotation is inspect.Parameter.empty:
        return False
    if callable(param.default) and not isinstance(param.default, type):
        return False
    return True


def _help_map(fn) -> "dict[str, str]":
    """Map parameter name → one-line help, parsed from ``fn``'s docstring."""
    if _parse_docstring is None or not fn.__doc__:
        return {}
    parsed = _parse_docstring(inspect.getdoc(fn) or "")
    out: dict[str, str] = {}
    for p in parsed.params:
        # Collapse multi-line descriptions to single lines — click's --help
        # column-wraps anyway, and pandorica's docstrings use indented
        # continuations that would otherwise show up as raw newlines.
        text = " ".join((p.description or "").split())
        if text:
            out[p.arg_name] = text
    return out


def _build_options(fn) -> "list[click.Option]":
    """Translate ``fn``'s annotated parameters into a list of click.Option."""
    sig = inspect.signature(fn)
    helps = _help_map(fn)
    options: list[click.Option] = []
    for name, p in sig.parameters.items():
        if not _is_exposable(p):
            continue
        ann = _unwrap_optional(p.annotation)
        has_default = p.default is not inspect.Parameter.empty
        default = p.default if has_default else None
        required = not has_default
        help_text = helps.get(name, "")
        flag_name = name.replace("_", "-")
        if ann is bool:
            # bool params (with or without ``Optional``) become paired flags.
            # ``default=None`` is honoured by click and lets pandorica's
            # tri-state knobs (e.g. ``use_gpu=None`` → auto) work without an
            # extra "auto" sentinel.
            decl = f"--{flag_name}/--no-{flag_name}"
            options.append(
                click.Option(
                    [decl],
                    default=default,
                    show_default=True,
                    help=help_text,
                )
            )
            continue
        options.append(
            click.Option(
                [f"--{flag_name}"],
                type=ann,
                default=default,
                required=required,
                show_default=True,
                help=help_text,
            )
        )
    return options


# ---------------------------------------------------------------------------
# Callback + command construction
# ---------------------------------------------------------------------------


_MISSING_DEP_MSG = (
    "The serial-section stitcher is not installed.\n"
    "Install with:\n"
    '    pip install "tardis_em[stitch]"\n'
    "or directly:\n"
    "    pip install pandorica docstring_parser"
)


def _run(**kwargs) -> None:
    """Invoke ``run_stitch`` with the parsed kwargs; show friendly errors."""
    if run_stitch is None:
        print_error(_MISSING_DEP_MSG, title="serial-section stitcher — package missing")
        raise SystemExit(1)
    try:
        run_stitch(**kwargs)
    except (FileNotFoundError, ValueError) as e:
        # Expected, user-facing problems (bad/empty folder, no usable sections,
        # nothing to stitch): show the message inside the TARDIS logo box, no
        # Python traceback.
        print_error(str(e), title="serial-section stitcher — cannot stitch")
        raise SystemExit(1)


def _short_description(fn) -> str:
    if _parse_docstring is not None and fn.__doc__:
        d = _parse_docstring(inspect.getdoc(fn) or "")
        if d.short_description:
            return d.short_description
    return "Stitch serial-section tomograms into one volume + merged graph."


def _build_command() -> click.Command:
    """Build the ``tardis_stitch`` click command at import time."""
    if run_stitch is None:
        # Without pandorica, ``--help`` and ``--version`` must still work so
        # users discover what's missing. Surface the install hint when the
        # command is actually run.
        @click.command("tardis_stitch", help=_MISSING_DEP_MSG)
        @click.version_option(version=version)
        def cmd() -> None:
            _run()

        return cmd

    cmd = click.Command(
        "tardis_stitch",
        params=_build_options(run_stitch),
        callback=_run,
        help=_short_description(run_stitch),
    )
    return click.version_option(version=version)(cmd)


main = _build_command()


if __name__ == "__main__":
    main()
