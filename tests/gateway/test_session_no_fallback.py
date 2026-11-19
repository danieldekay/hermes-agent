"""A ``no_fallback`` session runs its pinned route with NO provider fallback chain.

The session-create field (and its ``HERMES_NO_FALLBACK`` env equivalent) is worker-scoped: the
caller's own route loop (beads-worker's model allowlist) is the only fallback, while the profile's
``fallback_providers`` keep serving every other session unchanged (interactive chat keeps its GLM
fallback). Covers all three suppression points: the managed-worker ``construct_v1`` chain, the
in-process/reused agent chain, and the resolution-time AuthError walk.
"""
import json
from types import SimpleNamespace

import pytest

CHAIN = [{'provider': 'openrouter', 'model': 'z-ai/glm-5.3-flash'}]
LAUNCH = {'model': 'm', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1'}


def test_no_fallback_is_a_creation_field_riding_the_request(tmp_path):
    from gateway.session_policy import CREATE_FIELDS, build_policy
    # runtime.describe advertises CREATE_FIELDS | {'title}' as session_create parameters, so a
    # patched client and a patched daemon agree on the wire field (and an unpatched daemon
    # refuses the unknown option instead of silently keeping its chain).
    assert 'no_fallback' in CREATE_FIELDS
    policy = build_policy({'cwd': str(tmp_path), **LAUNCH, 'no_fallback': True}, {})
    assert policy.no_fallback is True
    assert json.loads(policy.request_json)['no_fallback'] is True
    ordinary = build_policy({'cwd': str(tmp_path), **LAUNCH}, {})
    assert ordinary.no_fallback is False


def test_no_fallback_passes_canonical_create_contract():
    from tui_gateway.contracts.registry import CANONICAL_METHODS, canonical_param_problems
    problems = canonical_param_problems(CANONICAL_METHODS['session.create'], {'no_fallback': True})
    assert problems == []


def test_env_launch_flags_maps_hermes_no_fallback(monkeypatch):
    from hermes_cli.gateway_chat import env_launch_flags
    monkeypatch.setenv('HERMES_NO_FALLBACK', '1')
    assert env_launch_flags().get('no_fallback') is True
    monkeypatch.delenv('HERMES_NO_FALLBACK')
    assert 'no_fallback' not in env_launch_flags()


def test_no_fallback_must_be_boolean(tmp_path):
    from gateway.session_policy import build_policy
    from hermes_state_runtime import RuntimeStoreError
    for bad in (1, 'true', None):
        with pytest.raises(RuntimeStoreError) as refused:
            build_policy({'cwd': str(tmp_path), **LAUNCH, 'no_fallback': bad}, {})
        assert refused.value.reason == 'invalid_params'


def _authority(tmp_path):
    runner = SimpleNamespace(config=None, _prefill_messages=None,
                             _refresh_fallback_model=lambda: list(CHAIN),
                             _get_system_prompt_for_channel=lambda *a: None)
    return SimpleNamespace(runner=runner, profile_id=str(tmp_path))


def test_managed_worker_construct_carries_no_chain_for_no_fallback(tmp_path):
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import construct_inputs
    authority = _authority(tmp_path)
    pinned = build_policy({'cwd': str(tmp_path), **LAUNCH, 'no_fallback': True},
                          {'fallback_providers': CHAIN})
    assert construct_inputs(authority, pinned, 'sid-1')['fallback_model'] is None
    ordinary = build_policy({'cwd': str(tmp_path), **LAUNCH}, {'fallback_providers': CHAIN})
    assert construct_inputs(authority, ordinary, 'sid-2')['fallback_model'] == CHAIN


def test_route_runtime_never_walks_the_chain_for_no_fallback(tmp_path, monkeypatch):
    from hermes_cli import runtime_provider
    import hermes_cli.runtime_provider_custom as runtime_custom
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import route_runtime

    sentinel = {'provider': 'openrouter', 'api_key': 'k', 'base_url': 'https://x', 'model': 'm'}
    walked = []
    monkeypatch.setattr(runtime_custom, '_resolve_named_custom_runtime', lambda **kw: None)
    monkeypatch.setattr(runtime_provider, 'resolve_runtime_provider', lambda **kw: sentinel)

    def _walk(config, **kw):
        walked.append(config)
        return sentinel, {'provider': 'openrouter', 'model': 'z-ai/glm-5.3-flash'}
    monkeypatch.setattr(runtime_provider, 'resolve_runtime_with_fallback', _walk)

    frozen = {'model': {'provider': 'openrouter', 'default': 'm'}}
    pinned = build_policy({'cwd': str(tmp_path), **LAUNCH, 'no_fallback': True}, {})
    runtime, entry = route_runtime(pinned, frozen, 'm', None)  # caller asks for the default walk
    assert runtime is sentinel and entry is None
    assert walked == [], 'a no_fallback policy must not consult fallback_providers on AuthError'
    ordinary = build_policy({'cwd': str(tmp_path), **LAUNCH}, {})
    runtime, entry = route_runtime(ordinary, frozen, 'm', None)
    assert walked == [frozen] and entry is not None


def _fresh_agent_kwargs(policy, monkeypatch, tmp_path):
    from gateway import session_policy
    from gateway.session_execution import GatewaySessionAgentMixin
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    built = {}
    turn = GatewaySessionAgentMixin.__new__(GatewaySessionAgentMixin)
    turn._runner = SimpleNamespace(session_authority=_authority(tmp_path), _prefill_messages=None,
                                   _service_tier=None, _session_db=None,
                                   _refresh_fallback_model=lambda: list(CHAIN))
    source = SimpleNamespace(chat_id='c', user_id='u', user_id_alt=None, user_name=None,
                             chat_name=None, chat_type='dm', thread_id=None)
    turn._ctx = SimpleNamespace(source=source, user_config={}, enabled_toolsets=[],
                                disabled_toolsets=None, session_id='c', session_key='k',
                                AIAgent=lambda **kw: built.update(kw))
    turn._build_fresh_agent({'model': 'm', 'runtime': {}}, 'local', None, 1, None, {}, False)
    return built


def test_in_process_turn_gets_no_chain_for_no_fallback(tmp_path, monkeypatch):
    from gateway.session_policy import build_policy
    pinned = build_policy({'cwd': str(tmp_path), **LAUNCH, 'no_fallback': True},
                          {'fallback_providers': CHAIN})
    assert _fresh_agent_kwargs(pinned, monkeypatch, tmp_path)['fallback_model'] is None
    ordinary = build_policy({'cwd': str(tmp_path), **LAUNCH}, {'fallback_providers': CHAIN})
    assert _fresh_agent_kwargs(ordinary, monkeypatch, tmp_path)['fallback_model'] == CHAIN


def test_reused_cached_agent_is_realigned_to_no_chain(tmp_path, monkeypatch):
    from gateway import session_execution, session_policy
    from gateway.session_execution import GatewaySessionAgentMixin
    from gateway.session_policy import build_policy

    pinned = build_policy({'cwd': str(tmp_path), **LAUNCH, 'no_fallback': True},
                          {'fallback_providers': CHAIN})
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: pinned)
    monkeypatch.setattr(session_execution, 'api_runtime_locked', lambda: None)
    applied = []
    agent = object()
    runner = SimpleNamespace(
        _agent_config_signature=lambda *a, **kw: 'sig', _agent_cache_lock=None, _agent_cache={},
        _extract_cache_busting_config=lambda cfg: None,
        _refresh_fallback_model=lambda: list(CHAIN),
        _apply_fallback_chain_to_agent=lambda ag, chain: applied.append(chain))
    turn = GatewaySessionAgentMixin.__new__(GatewaySessionAgentMixin)
    turn._runner = runner
    turn._ctx = SimpleNamespace(source=SimpleNamespace(chat_id='c', user_id='u', user_id_alt=None,
                                                       user_name=None, chat_name=None,
                                                       chat_type='dm', thread_id=None),
                                user_config={}, enabled_toolsets=[], disabled_toolsets=None,
                                session_id='c', session_key='k')
    monkeypatch.setattr(GatewaySessionAgentMixin, '_cached_sid_is_dead',
                        lambda self, lock, cache: ('sid', False))
    monkeypatch.setattr(GatewaySessionAgentMixin, '_current_message_count', lambda self: 0)
    monkeypatch.setattr(GatewaySessionAgentMixin, '_lookup_cached_agent',
                        lambda self, *a, **kw: SimpleNamespace(agent=agent, reused=True, evicted=None))
    got, reused = turn._resolve_turn_agent({'model': 'm', 'runtime': {}}, 'local', None, 1, None, {})
    assert got is agent and reused is True
    assert applied == [None], 'a reused agent of a no_fallback session must end with an empty chain'
