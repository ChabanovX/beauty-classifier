from http import HTTPStatus
from fastapi.testclient import TestClient
from typing import Any, NamedTuple
from unittest.mock import AsyncMock

from pytest_cases import parametrize_with_cases

from src.interfaces.api.v1.__main__ import create_app

# <obj_to_mock> -> <<attribute_to_mock> -> <side_effect_fn>>
MockMappingT = dict[Any, dict[str, Any]]


class APICase(NamedTuple):
    """Полная репрезентация тестового кейса апи

    Определяет эндпоинт, который надо тестировать,
    входные данные, которые передадутся (*args и **kwargs),
    ожидаемый статус код и результат.

    `mocks` определяет, какие аттрибуты каких объектов
    надо замокать и чем.
    """

    login: tuple[str, str] = ("", "")
    inp_body: dict[str, Any] = {}
    inp_headers: dict[str, Any] = {}
    endpoint_to_test: str = "/"
    method: str = "GET"
    mocks: MockMappingT = {}
    expected_status: HTTPStatus = HTTPStatus.OK
    expected_body: dict[str, Any] | list | str | bytes | None = {}
    expected_headers: dict[str, Any] = {}


def create_api_test(cases: Any):
    """
    Создает отдельную тестовую функцию из кейсов
    """

    @parametrize_with_cases(
        argnames=APICase._fields,
        cases=cases,
    )
    def test_api_func(
        login: tuple[str, str],
        inp_body: dict[str, Any],
        inp_headers: dict[str, Any],
        endpoint_to_test: str,
        method: str,
        mocks: MockMappingT,
        expected_status: HTTPStatus,
        expected_body: dict[str, Any] | list | str | bytes | None,
        expected_headers: dict[str, Any],
        monkeypatch,
    ):
        """
        Тестовая функция.

        Args:
        ----
            Должны совпадать с полями класса APICase.
            Другие дополнительные фикстуры.

        """
        for dependency, attrs_to_mock in mocks.items():
            for attr_name, side_effect in attrs_to_mock.items():
                monkeypatch.setattr(
                    dependency,
                    attr_name,
                    AsyncMock(side_effect=side_effect),
                )
        client = get_client(login[0], login[1])
        client.headers.update(inp_headers)
        print(client.headers.get("Authorization", None))
        res = client.request(
            method,
            endpoint_to_test,
            json=inp_body,
        )
        assert res.status_code == expected_status, (
            f"{res.status_code} != {expected_status}, {res.json()}"
        )
        if not expected_status.is_success:
            assert "detail" in res.json() or "loc" in res.json()
        elif expected_body:
            try:
                assert res.json() == expected_body, f"{res.json()} != {expected_body}"
            except ValueError:
                # if not a json (string or empty)
                assert res.content == bytes(expected_body), (
                    f"{res.content} != {bytes(expected_body)}"
                )
        if expected_headers:
            assert res.headers == expected_headers, (
                f"{res.headers} != {expected_headers}"
            )

    return test_api_func


def get_client(name: str = "", password: str = ""):
    with TestClient(create_app()) as client:
        if not (name and password):
            return client
        res = client.post("/auth/register", json={"name": name, "password": password})
        token = res.json().get("token", None)
        if token is None:
            res = client.post("/auth/login", json={"name": name, "password": password})
            token = res.json().get("token", None)
        res.raise_for_status()
        client.headers.update({"Authorization": f"Bearer {token['token']}"})
        return client
