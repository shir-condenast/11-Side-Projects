import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


class StateAdapter:
    def __init__(self, backend):
        self._backend = backend

    def exists(self, key):
        return key in self._backend

    def get(self, key, default=None):
        return self._backend.get(key, default)

    def set(self, key, value):
        self._backend[key] = value


class DebugState(dict):
    pass


def is_streamlit():
    return get_script_run_ctx() is not None


def get_state():
    if is_streamlit():
        return StateAdapter(st.session_state)

    global _DEBUG_STATE
    if "_DEBUG_STATE" not in globals():
        _DEBUG_STATE = DebugState()

    return StateAdapter(_DEBUG_STATE)
