#!/usr/bin/env python


import re
from datetime import timedelta

import cftime
from compliance_checker.base import BaseCheck, TestCtx

from checks.utils import deltdic, severity_word


def check_time_chunking(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if the chunking with respect to the time dimension is in accordance with CORDEX-CMIP6 Archive Specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXT001"
    desc = f"[{check_id}] Time Chunking"
    testctx = TestCtx(severity, desc)
    failure_registered = False

    # Check if frequency is known and supported
    # Supported is the intersection of:
    #  CORDEX-CMIP6: fx, 1hr, day, mon
    #  deltdic.keys() - whatever frequencies are defined there
    if CheckerObject.frequency in ["unknown", "fx"]:
        testctx.add_pass()
        return [testctx.to_result()]
    if CheckerObject.frequency not in [
        "1hr",
        "6hr",
        "day",
        "mon",
    ]:
        testctx.add_failure(
            f"Frequency '{CheckerObject.frequency}' not supported by this check."
        )
        return [testctx.to_result()]

    # Expected last simulation year
    # -> suppress error for last year of a simulation
    #    if it is the "official" last year
    expected_last_sim_year = {
        "evaluation": 2024,
        "historical": 2014,
        "ssp119": 2100,
        "ssp126": 2100,
        "ssp245": 2100,
        "ssp370": 2100,
        "ssp585": 2100,
    }

    # Get the time dimension, calendar and units
    if CheckerObject.time is None:
        testctx.add_failure("Coordinate variable 'time' not found in file.")
        return [testctx.to_result()]
    if CheckerObject.calendar is None:
        testctx.add_failure("'time' variable has no 'calendar' attribute.")
        failure_registered = True
    if CheckerObject.timeunits is None:
        testctx.add_failure("'time' variable has no 'units' attribute.")
        failure_registered = True
    if failure_registered:
        return [testctx.to_result()]

    # Get the first and last time values
    first_time = CheckerObject.time[0].values
    last_time = CheckerObject.time[-1].values

    # Convert the first and last time values to cftime.datetime objects
    first_time = cftime.num2date(
        first_time, calendar=CheckerObject.calendar, units=CheckerObject.timeunits
    )
    last_time = cftime.num2date(
        last_time, calendar=CheckerObject.calendar, units=CheckerObject.timeunits
    )

    # Calculate the expected start and end dates of the year
    expected_start_date = cftime.datetime(
        first_time.year, 1, 1, 0, 0, 0, calendar=CheckerObject.calendar
    )
    if CheckerObject.frequency in ["1hr", "6hr"]:
        expected_end_date = cftime.datetime(
            first_time.year + 1, 1, 1, 0, 0, 0, calendar=CheckerObject.calendar
        )
    elif CheckerObject.frequency == "day":
        year = first_time.year + 1
        while str(year)[-1] not in ["1", "6"]:
            year += 1
        expected_end_date = cftime.datetime(
            year, 1, 1, 0, 0, 0, calendar=CheckerObject.calendar
        )
    else:
        year = first_time.year + 1
        while str(year)[-1] != "1":
            year += 1
        expected_end_date = cftime.datetime(
            year, 1, 1, 0, 0, 0, calendar=CheckerObject.calendar
        )

    # File chunks as requested by CORDEX-CMIP6
    if CheckerObject.frequency == "mon":
        nyears = 10
    elif CheckerObject.frequency == "day":
        nyears = 5
    # subdaily
    else:
        nyears = 1

    # Apply calendar- and frequency-dependent adjustments
    offset = 0
    if CheckerObject.calendar == "360_day" and CheckerObject.frequency == "mon":
        offset = timedelta(hours=12)

    # Modify expected start and end dates based on cell_methods and above offset
    if bool(
        re.fullmatch("^.*time: point.*$", CheckerObject.cell_methods, flags=re.ASCII)
    ):
        expected_end_date = expected_end_date - timedelta(
            seconds=deltdic[CheckerObject.frequency] - offset - offset
        )
    elif bool(
        re.fullmatch(
            "^.*time: (maximum|minimum|mean|sum).*$",
            CheckerObject.cell_methods,
            flags=re.ASCII,
        )
    ):
        expected_start_date += timedelta(
            seconds=deltdic[CheckerObject.frequency] / 2.0 - offset
        )
        expected_end_date -= timedelta(
            seconds=deltdic[CheckerObject.frequency] / 2.0 - offset
        )
    else:
        testctx.add_failure(
            f"Cannot interpret cell_methods '{CheckerObject.cell_methods}'."
        )
        return [testctx.to_result()]

    # Consider experiment end
    exp_last_year = expected_last_sim_year.get(
        CheckerObject._get_attr("driving_experiment_id", None), None
    )
    if exp_last_year:
        expected_end_date_exp = expected_end_date.replace(year=exp_last_year)
    else:
        expected_end_date_exp = expected_end_date

    errmsg = (
        f"File chunking {severity_word(severity, noun=True)}s: "
        f"{'Apart from the first and last files of a timeseries ' if nyears>1 else ''}"
        f"'{nyears}' full simulation year{' is' if nyears==1 else 's are'} "
        f"expected in the data file for frequency '{CheckerObject.frequency}'."
    )
    # Check if the first time is equal to the expected start date
    if first_time != expected_start_date:
        testctx.add_failure(
            errmsg
            + f" The first timestep conflicts with this {severity_word(severity, noun=True)} ('{expected_start_date}'): '{first_time}'. "
        )
        failure_registered = True
    # Check if the last time is equal to the expected end date
    if last_time != expected_end_date and last_time != expected_end_date_exp:
        testctx.add_failure(
            errmsg
            + f" The last timestep conflicts with this {severity_word(severity, noun=True)} ('{expected_end_date}'): '{last_time}'. "
        )
        failure_registered = True

    if not failure_registered:
        testctx.add_pass()

    return [testctx.to_result()]


def check_time_range(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if the time range is as expected according to the CORDEX-CMIP6 Archive Specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXT002"
    desc = f"[{check_id}] Time Range"
    testctx = TestCtx(severity, desc)
    failure_registered = False

    # Check if frequency is known and supported
    if CheckerObject.frequency in ["unknown", "fx"]:
        # Potential error would be raised in another check
        testctx.add_pass()
        return [testctx.to_result()]
    if CheckerObject.frequency not in [
        "1hr",
        "6hr",
        "day",
        "mon",
    ]:
        testctx.add_failure(
            f"Frequency '{CheckerObject.frequency}' not supported by this check."
        )
        return [testctx.to_result()]

    # Get the time dimension, calendar and units
    if CheckerObject.time is None:
        testctx.add_failure("Coordinate variable 'time' not found in file.")
        return [testctx.to_result()]
    if CheckerObject.calendar is None:
        testctx.add_failure("'time' variable has no 'calendar' attribute.")
        failure_registered = True
    if CheckerObject.timeunits is None:
        testctx.add_failure("'time' variable has no 'units' attribute.")
        failure_registered = True
    if failure_registered:
        return [testctx.to_result()]

    # Get the first and last time values
    first_time = CheckerObject.time[0].values
    last_time = CheckerObject.time[-1].values

    # Convert the first and last time values to cftime.datetime objects
    first_time = cftime.num2date(
        first_time, calendar=CheckerObject.calendar, units=CheckerObject.timeunits
    )
    last_time = cftime.num2date(
        last_time, calendar=CheckerObject.calendar, units=CheckerObject.timeunits
    )

    # Compile the expected time_range
    if CheckerObject.frequency == "mon":
        time_range = f"{first_time.strftime(format='%4Y%2m')}-{last_time.strftime(format='%4Y%2m')}"
    elif CheckerObject.frequency == "day":
        time_range = f"{first_time.strftime(format='%4Y%2m%2d')}-{last_time.strftime(format='%4Y%2m%2d')}"
    elif CheckerObject.frequency in ["1hr", "6hr"]:
        time_range = f"{first_time.strftime(format='%4Y%2m%2d%2H%2M')}-{last_time.strftime(format='%4Y%2m%2d%2H%2M')}"
    else:
        time_range = ""

    # Check if the time_range is as expected
    if CheckerObject.drs_fn["time_range"] != time_range:
        testctx.add_failure(
            f"Expected time_range '{time_range}' but found '"
            f"{CheckerObject.drs_fn['time_range'] if CheckerObject.drs_fn['time_range'] else 'unset'}'."
        )
        failure_registered = True

    if not failure_registered:
        testctx.add_pass()

    return [testctx.to_result()]


def check_calendar(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if the time attributes are as recommended.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXT003"
    desc = f"[{check_id}] Calendar"
    testctx = TestCtx(severity, desc)

    # Check calendar
    if CheckerObject.time is not None:
        if CheckerObject.calendar not in ["standard", "proleptic_gregorian"]:
            msg = (
                f"Your 'calendar' attribute is not one of the {severity_word(severity)} calendars "
                f"('standard', 'proleptic_gregorian'): '{CheckerObject.calendar}'."
            )
            if CheckerObject.calendar == "gregorian":
                msg += (
                    " Please use the 'standard' calendar, since the use of the 'gregorian' "
                    "calendar is deprecated since CF-1.9."
                )
            else:
                msg += " The use of another calendar is OK in case it has been inherited from the driving model."
            testctx.add_failure(msg)
        else:
            testctx.add_pass()
    else:
        testctx.add_pass()

    return [testctx.to_result()]


def check_time_units(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if the time units are as requested.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXT004"
    desc = f"[{check_id}] Time Units"
    testctx = TestCtx(severity, desc)

    if CheckerObject.time is not None:
        if CheckerObject.timeunits not in [
            "days since 1950-01-01T00:00:00Z",
            "days since 1950-01-01T00:00:00",
            "days since 1950-01-01 00:00:00",
            "days since 1950-01-01",
            "days since 1950-1-1",
            "days since 1850-01-01T00:00:00Z",
            "days since 1850-01-01T00:00:00",
            "days since 1850-01-01 00:00:00",
            "days since 1850-01-01",
            "days since 1850-1-1",
        ]:
            testctx.add_failure(
                "Your time axis' 'units' attribute differs from the allowed values "
                "('days since 1950-01-01T00:00:00Z', 'days since 1850-01-01T00:00:00Z' "
                f"if the pre-1950 era is included in the group's simulations): '{CheckerObject.timeunits}'."
            )
        else:
            testctx.add_pass()
    else:
        testctx.add_pass()

    return [testctx.to_result()]
