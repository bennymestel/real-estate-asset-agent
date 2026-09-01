import pytest

from src.data.catalog import get_catalog
from src.data.resolver import resolve_ledger_category, resolve_property, resolve_tenant


@pytest.mark.parametrize("query", [
    "Building 120", "building 120", "building120", "b120", "B120",
    "bldg 120", "120", "buildng 120",
])
def test_property_variants_resolve_to_canonical(query):
    r = resolve_property(query)
    assert r.value == "Building 120", (query, r.score)


def test_building_17_is_not_confused_with_170():
    assert resolve_property("Building 17").value == "Building 17"
    assert resolve_property("17").value == "Building 17"


@pytest.mark.parametrize("query", ["Tenant 7", "tenant 7", "tenant7", "t7", "T7"])
def test_tenant_variants_resolve(query):
    assert resolve_tenant(query).value == "Tenant 7"


def test_unknown_property_is_unresolved_with_candidates():
    r = resolve_property("the Oak building")
    assert r.value is None
    assert set(r.candidates) <= set(get_catalog().properties)


def test_nonexistent_building_number_is_unresolved():
    r = resolve_property("Building 999")
    assert r.value is None


def test_ledger_category_from_plain_english():
    assert resolve_ledger_category("bank charges").value == "bank_charges"
    assert resolve_ledger_category("real estate taxes").value == "real_estate_taxes"
