from .square_table import (
    SquareTable,
    OneLeg,
    JustOneLeg,
    TableWithOneLeg,
    SquareTablePatchFix,
)


def furniture_factory(furniture: str, seed: int):
    if furniture == "square_table":
        return SquareTable(seed)
    elif furniture == "one_leg":
        return OneLeg(seed)
    elif furniture == "just_one_leg":
        return JustOneLeg(seed)
    elif furniture == "table_with_one_leg":
        return TableWithOneLeg(seed)
    elif furniture == "square_table_patch_fix":
        return SquareTablePatchFix(seed)
    else:
        raise ValueError(f"Unknown furniture type: {furniture}")
