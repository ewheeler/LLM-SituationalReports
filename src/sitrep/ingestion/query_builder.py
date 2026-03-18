from __future__ import annotations

from datetime import datetime, time, timezone

from sitrep.types import EventQuery


RELIEFWEB_FIELDS = (
    "title",
    "body",
    "body-html",
    "date.created",
    "language.name",
    "source.name",
    "source.shortname",
    "country.iso3",
    "country.name",
    "file.url",
    "file.filename",
    "file.mimetype",
    "format.name",
    "theme.name",
    "disaster.name",
    "url",
    "url_alias",
)


def build_keyword_query(event_query: EventQuery) -> str:
    terms = [event_query.event_name, *event_query.keywords]
    return " ".join(term for term in terms if term).strip()


def build_reliefweb_request_params(
    event_query: EventQuery,
    page_size: int,
    offset: int = 0,
) -> dict[str, object]:
    conditions: list[dict[str, object]] = [
        {
            "field": "date.created",
            "value": {
                "from": _iso_datetime_start(event_query.date_from),
                "to": _iso_datetime_end(event_query.date_to),
            },
        }
    ]
    if event_query.country_codes:
        conditions.append(
            {
                "field": "country.iso3",
                "value": list(event_query.country_codes),
                "operator": "OR",
            }
        )

    payload: dict[str, object] = {
        "preset": "latest",
        "limit": page_size,
        "offset": offset,
        "sort": ["date.created:desc"],
        "fields": {"include": list(RELIEFWEB_FIELDS)},
        "filter": {
            "operator": "AND",
            "conditions": conditions,
        },
    }
    keyword_query = build_keyword_query(event_query)
    if keyword_query:
        payload["query"] = {"value": keyword_query}
    return payload


def build_rss_request_params(event_query: EventQuery) -> dict[str, object]:
    return {
        "query": build_keyword_query(event_query),
        "date_from": event_query.date_from.isoformat(),
        "date_to": event_query.date_to.isoformat(),
        "countries": list(event_query.country_codes),
    }


def _iso_datetime_start(value):
    return datetime.combine(value, time.min, tzinfo=timezone.utc).isoformat()


def _iso_datetime_end(value):
    return datetime.combine(value, time.max.replace(microsecond=0), tzinfo=timezone.utc).isoformat()
