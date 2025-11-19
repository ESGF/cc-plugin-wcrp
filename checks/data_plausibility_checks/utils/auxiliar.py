import os
from compliance_checker.base import TestCtx

def dump_data_file(dataset, variable, check_name, ctx):
    """
    Dumps the results of a check to a file.
    Parameters:
    - dataset: netCDF4.Dataset or similar object, the dataset being checked.
    - variable: str, the name of the variable being checked.
    - check_name: str, the name of the check being performed.
    - ctx: TestCtx, the context containing messages and results of the check.
    """
    print ("Dumping results to file...")
    try:
        dataset_path = dataset.filepath() if hasattr(dataset, "filepath") else str(variable)
    except Exception:
        dataset_path = str(variable)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(
        reports_dir,
        f"{dataset_name}_{check_name}_fail.txt"
    )
    with open(filename, "w") as f:
        f.write(f"Test: {check_name}\n")
        f.write(f"File: {dataset_path}\n")
        f.write("Messages:\n")
        for msg in ctx.messages:
            f.write(f"{msg}\n")
        if hasattr(ctx, "failures"):
            f.write("Failures:\n")
            for fail in ctx.failures:
                f.write(f"{fail}\n")
        f.write(f"Score: {ctx.score}\n")

def dump_data_file_extended(dataset, variable, check_name, ctx, options=''):
    """
    Dumps the results of a check to a file (versión extendida con coordenadas legibles).
    """

    try:
        dataset_path = dataset.filepath() if hasattr(dataset, "filepath") else str(variable)
    except Exception:
        dataset_path = str(variable)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(
        reports_dir,
        f"{dataset_name}_{check_name}_{options}_fail.txt"
    )

    with open(filename, "w") as f:
        f.write("==== Test Report ====\n")
        f.write(f"Test: {check_name}\n")
        f.write(f"File: {dataset_path}\n")
        f.write(f"Variable: {ctx.variable}\n")
        f.write(f"Dataset Name: {ctx.dataset_name}\n")
        f.write(f"Parameters: {ctx.parameters}\n")
        # f.write(f"Score: {ctx.score}/{len(ctx.passes) + len(ctx.failures)}\n\n")
        f.write("--- Messages ---\n")
        for msg in ctx.messages:
            f.write(msg + "\n")
        f.write("\n")


        f.write("--- Results (coordinates, values) ---\n")
        for coord in ctx.coordinates[:5]:
            f.write(f"{coord}\n")
        if len(ctx.coordinates)>5:
            f.write("The rest of the coordinates are in the summary\n")
        f.write("--- Summary ---\n")
        f.write(ctx.summarize() + "\n")


class Coordinate:
    def __init__(self, name, indices=None, values=None, result=None):
        self.name = name
        self.indices = indices if indices is not None else []
        self.values = values if values is not None else []
        self.result = result  # atributo usado en contextos de prueba

        # Para compatibilidad con coordinates_to_string
        if isinstance(values, dict):
            self.values_dict = values
        else:
            if self.indices and self.values:
                self.values_dict = {i: v for i, v in zip(self.indices, self.values)}
            else:
                self.values_dict = {"value": self.values}

    def __str__(self):
        lines = []
        for i, val in zip(self.indices, self.values):
            lines.append(f"{self.name}\t{i}\t{val}")
        return "\n".join(lines)


class ExtendedTestCtx(TestCtx):
    def __init__(self, category=None, description="", out_of=0, score=0,
                 messages=None, variable=None, dataset_name=None,
                 test_function=None, parameters=None):
        super().__init__(category, description, out_of, score, messages, variable)
        self.dataset_name = dataset_name
        self.test_function = test_function
        self.parameters = parameters or {}
        self.coordinates = []  # lista de objetos Coordinate

    def add_coordinate(self, name, indices=None, values=None, result_value=None):
        coord = Coordinate(name, indices=indices, values=values, result=result_value)
        self.coordinates.append(coord)


    def coordinates_to_string(self):
        if not self.coordinates:
            return "No coordinates stored."
        rows = []
        for coord in self.coordinates:
            for idx, val_score in zip(coord.indices, coord.values):
                if ',' in val_score:
                    val, score = val_score.split(',', 1)  # Split on first comma only
                    rows.append(f"{coord.name}\t{idx}\t{val}\t{score}")
                else:
                    rows.append(f"{coord.name}\t{idx}\t{val_score}")
        return "Name\tIndex\tValue\tScore\n" + "\n".join(rows)

    def summarize(self):
        summary = (
            f"Dataset: {self.dataset_name}\n"
            f"Test: {self.test_function}\n"
            f"Variable: {self.variable}\n"
            f"Parameters: {self.parameters}\n"
            f"Score: {self.score}/{self.out_of}\n"
            f"Messages: {self.messages}\n"
        )
        if self.coordinates:
            summary += "Coordinates and results:\n"
            summary += self.coordinates_to_string()
        return summary