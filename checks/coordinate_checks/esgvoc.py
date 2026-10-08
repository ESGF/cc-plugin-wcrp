"""Read coordinate metadata from ESGVoc with a strict compatibility gate."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

from checks.coordinate_checks.model import (
    Catalog,
    as_dict,
    records_by_id,
    reference_ids,
)

MINIMUM_ESGVOC_VERSION = Version("7.0.0")

DATA_COORDINATE_FIELDS = [
    "id",
    "coordinate_type",
    "axis",
    "data_type",
    "long_name",
    "cf_standard_name",
    "out_name",
    "units",
    "positive",
    "stored_direction",
    "coordinate_values",
    "coordinate_bounds",
    "bounds_required",
    "tolerance",
    "valid_min",
    "valid_max",
    "is_climatology",
    "is_generic_model_level_coordinate",
]
MODEL_LEVEL_FIELDS = [
    "id",
    "axis",
    "data_type",
    "long_name",
    "cf_standard_name",
    "computed_standard_name",
    "out_name",
    "units",
    "positive",
    "stored_direction",
    "bounds_required",
    "valid_min",
    "valid_max",
    "formula",
    "z_factors",
    "z_bounds_factors",
    "generic_level_name",
]
FORMULA_TERM_FIELDS = [
    "id",
    "data_type",
    "long_name",
    "cf_standard_name",
    "out_name",
    "units",
    "dimensions",
]
GRID_VARIABLE_FIELDS = [
    "id",
    "data_type",
    "long_name",
    "cf_standard_name",
    "out_name",
    "units",
    "dimensions",
    "valid_min",
    "valid_max",
]
GRID_AXIS_FIELDS = [
    "id",
    "axis",
    "data_type",
    "long_name",
    "cf_standard_name",
    "out_name",
    "units",
]


class CoordinateMetadataError(RuntimeError):
    """The coordinate standard could not be initialized reliably."""


def require_supported_version(installed_version: str | None = None) -> str:
    if installed_version is None:
        try:
            installed_version = version("esgvoc")
        except PackageNotFoundError as exc:
            raise CoordinateMetadataError(
                f"Coordinate checks require esgvoc>={MINIMUM_ESGVOC_VERSION}, "
                "but the 'esgvoc' "
                "package is not installed. Install it separately before running "
                "the CMIP7 coordinate checks."
            ) from exc
    try:
        parsed = Version(installed_version)
    except InvalidVersion as exc:
        raise CoordinateMetadataError(
            f"Coordinate checks require esgvoc>={MINIMUM_ESGVOC_VERSION}, but "
            "the installed version "
            f"{installed_version!r} could not be interpreted."
        ) from exc
    if parsed < MINIMUM_ESGVOC_VERSION:
        raise CoordinateMetadataError(
            f"Coordinate checks require esgvoc>={MINIMUM_ESGVOC_VERSION} "
            "because earlier releases do not provide the required coordinate "
            "descriptor models; found "
            f"esgvoc=={installed_version}."
        )
    return installed_version


def _get_api():
    try:
        import esgvoc.api as api
    except ImportError as exc:
        raise CoordinateMetadataError(
            f"Coordinate checks require esgvoc>={MINIMUM_ESGVOC_VERSION}, but "
            "ESGVoc could not be "
            f"imported: {type(exc).__name__}: {exc}"
        ) from exc
    require_supported_version()
    return api


def _all(api, project_id: str, descriptor: str, fields: list[str]):
    try:
        records = api.get_all_terms_in_collection(project_id, descriptor, fields)
    except Exception as exc:
        raise CoordinateMetadataError(
            f"ESGVoc could not read project {project_id!r} '{descriptor}' "
            "coordinate metadata: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not records:
        raise CoordinateMetadataError(
            f"ESGVoc returned an empty result for project {project_id!r} "
            f"'{descriptor}' "
            "collection; the coordinate catalog is incomplete."
        )
    return records


def _all_with_universe_fallback(
    api, project_id: str, descriptor: str, fields: list[str]
):
    """Read project records, falling back to the matching Universe descriptor."""
    project_error = None
    try:
        records = api.get_all_terms_in_collection(project_id, descriptor, fields)
    except Exception as exc:
        project_error = exc
        records = []
    if records:
        return records
    try:
        records = api.get_all_terms_in_data_descriptor(descriptor, fields)
    except Exception as exc:
        project_detail = (
            f" Project collection error: {type(project_error).__name__}: "
            f"{project_error}."
            if project_error is not None
            else ""
        )
        raise CoordinateMetadataError(
            f"ESGVoc could not read coordinate metadata {descriptor!r} from "
            f"project {project_id!r} or its Universe data descriptor."
            f"{project_detail} Universe descriptor error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not records:
        raise CoordinateMetadataError(
            "ESGVoc returned no records for coordinate collection/data "
            f"descriptor {descriptor!r}; the coordinate catalog is incomplete."
        )
    return records


def load_catalog(
    branded_variable_id: str,
    *,
    project_id: str = "cmip7",
    api=None,
    installed_version: str | None = None,
    branded_collection: str = "branded_variable",
    file_variable_name: str = "",
    allow_universe_coordinate_fallback: bool = False,
) -> Catalog:
    """Read the branded variable and complete coordinate catalog exactly once."""
    if api is None:
        api = _get_api()
    else:
        require_supported_version(installed_version)

    if not branded_variable_id:
        raise CoordinateMetadataError(
            "The file has no non-empty global 'branded_variable' attribute, so "
            "its required coordinate IDs cannot be obtained from ESGVoc."
        )
    lookup_id = str(branded_variable_id).strip().lower()
    try:
        branded_record = api.get_term_in_collection(
            project_id,
            branded_collection,
            lookup_id,
            ["id", "out_name", "variable_root_name", "dimensions", "cell_methods"],
        )
    except Exception as exc:
        raise CoordinateMetadataError(
            f"ESGVoc failed while reading {branded_collection} "
            f"{branded_variable_id!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if branded_record is None:
        raise CoordinateMetadataError(
            f"Known branded variable {branded_variable_id!r} was not found in "
            f"ESGVoc project collection {project_id!r}/{branded_collection!r}."
        )
    branded = as_dict(branded_record)
    coordinate_ids = reference_ids(branded.get("dimensions"))
    if not coordinate_ids:
        raise CoordinateMetadataError(
            f"Known branded variable {branded_variable_id!r} has no usable ordered "
            "coordinate references in ESGVoc field 'dimensions'."
        )

    read_all = (
        _all_with_universe_fallback if allow_universe_coordinate_fallback else _all
    )
    data_coordinates = records_by_id(
        read_all(api, project_id, "data_coordinate", DATA_COORDINATE_FIELDS)
    )
    missing = [
        identifier
        for identifier in coordinate_ids
        if identifier not in data_coordinates
    ]
    if missing:
        raise CoordinateMetadataError(
            f"Known branded variable {branded_variable_id!r} references coordinate "
            f"ID(s) absent from project {project_id!r} data_coordinate collection: "
            f"{missing}."
        )

    return Catalog(
        project_id=project_id,
        branded_variable_id=str(branded.get("id") or lookup_id),
        branded_variable=branded,
        coordinate_ids=tuple(coordinate_ids),
        data_coordinates=data_coordinates,
        model_levels=records_by_id(
            read_all(api, project_id, "model_level_coordinate", MODEL_LEVEL_FIELDS)
        ),
        formula_terms=records_by_id(
            read_all(api, project_id, "formula_term", FORMULA_TERM_FIELDS)
        ),
        grid_variables=records_by_id(
            read_all(api, project_id, "grid_variable", GRID_VARIABLE_FIELDS)
        ),
        grid_axes=records_by_id(
            read_all(api, project_id, "grid_axis", GRID_AXIS_FIELDS)
        ),
        file_variable_name=file_variable_name,
    )
