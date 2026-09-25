"""Tests for `validate_call(validate_return=True)` on async generator functions.

Each yielded value is validated against the item type and each non-priming
`asend()` value against the send type, one at a time, as iteration happens.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator

import pytest

from pydantic import ValidationError, validate_call


def make_traced(log: list[str]):
    """An async generator recording its execution order into `log`."""

    @validate_call(validate_return=True)
    async def traced(n: int) -> AsyncGenerator[int, str]:
        log.append('body:start')
        try:
            for i in range(n):
                log.append(f'yield:{i}')
                cmd = yield i
                log.append(f'cmd:{cmd}')
        finally:
            log.append('body:finally')

    return traced


def test_async_gen_normal_iteration_records_order():
    log: list[str] = []
    traced = make_traced(log)

    async def run():
        gen = traced(3)
        received = [await gen.__anext__()]
        received.append(await gen.asend('a'))
        received.append(await gen.asend('b'))
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
        return received

    assert asyncio.run(run()) == [0, 1, 2]
    assert log == [
        'body:start',
        'yield:0',
        'cmd:a',
        'yield:1',
        'cmd:b',
        'yield:2',
        'cmd:None',
        'body:finally',
    ]


def test_async_gen_created_lazily():
    log: list[str] = []
    traced = make_traced(log)

    gen = traced(2)
    # Calling the function validates the arguments but does not run the body:
    assert log == []

    async def run():
        assert await gen.__anext__() == 0
        await gen.aclose()

    asyncio.run(run())
    assert log == ['body:start', 'yield:0', 'body:finally']


def test_async_gen_arguments_validated_at_call_time():
    log: list[str] = []
    traced = make_traced(log)

    with pytest.raises(ValidationError) as exc_info:
        traced('not-an-int')
    assert exc_info.value.errors()[0]['loc'] == (0,)
    assert log == []


def test_async_gen_invalid_send_reports_send_index():
    log: list[str] = []
    traced = make_traced(log)

    async def run():
        gen = traced(3)
        assert await gen.__anext__() == 0
        assert await gen.asend('ok') == 1
        with pytest.raises(ValidationError) as exc_info:
            await gen.asend(123)  # not a valid `str`
        # The generator is untouched by the rejected value and keeps working:
        assert await gen.asend('still-alive') == 2
        await gen.aclose()
        return exc_info.value

    exc = asyncio.run(run())
    assert exc.title == 'make_traced.<locals>.traced send #2'
    assert exc.errors()[0]['loc'] == ('send', 2)
    # The invalid value never reached the generator body:
    assert log == ['body:start', 'yield:0', 'cmd:ok', 'yield:1', 'cmd:still-alive', 'yield:2', 'body:finally']


def test_async_gen_invalid_yield_reports_yield_index():
    log: list[str] = []

    @validate_call(validate_return=True)
    async def gen() -> AsyncGenerator[int, str]:
        try:
            yield 1
            log.append('about-to-yield-bad')
            yield 'not-an-int'
        finally:
            log.append('body:finally')

    async def run():
        g = gen()
        assert await g.__anext__() == 1
        with pytest.raises(ValidationError) as exc_info:
            await g.__anext__()
        return exc_info.value

    exc = asyncio.run(run())
    assert exc.title == 'test_async_gen_invalid_yield_reports_yield_index.<locals>.gen yield #2'
    assert exc.errors()[0]['loc'] == ('yield', 2)
    assert log == ['about-to-yield-bad', 'body:finally']


def test_async_gen_yield_values_are_coerced():
    @validate_call(validate_return=True)
    async def gen() -> AsyncGenerator[int, str]:
        received = yield '1'  # coerced to `int` on the way out
        yield len(received)  # `received` is the coerced `str` command

    async def run():
        g = gen()
        assert await g.__anext__() == 1
        assert await g.asend(b'abc') == 3  # coerced to `str` on the way in

    asyncio.run(run())


def test_async_gen_athrow_triggers_finally():
    log: list[str] = []
    traced = make_traced(log)

    async def run():
        gen = traced(3)
        assert await gen.__anext__() == 0
        with pytest.raises(RuntimeError, match='boom'):
            await gen.athrow(RuntimeError('boom'))

    asyncio.run(run())
    assert log == ['body:start', 'yield:0', 'body:finally']


def test_async_gen_athrow_caught_yield_is_validated():
    @validate_call(validate_return=True)
    async def gen() -> AsyncGenerator[int, str]:
        try:
            yield 1
        except ValueError:
            yield 'not-an-int'  # recovered, but the yielded value is still validated

    async def run():
        g = gen()
        assert await g.__anext__() == 1
        with pytest.raises(ValidationError) as exc_info:
            await g.athrow(ValueError)
        assert exc_info.value.errors()[0]['loc'] == ('yield', 2)

    asyncio.run(run())


def test_async_gen_aclose_triggers_finally():
    log: list[str] = []
    traced = make_traced(log)

    async def run():
        gen = traced(3)
        assert await gen.__anext__() == 0
        await gen.aclose()
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    asyncio.run(run())
    assert log == ['body:start', 'yield:0', 'body:finally']


def test_async_gen_cancellation_triggers_finally():
    log: list[str] = []

    @validate_call(validate_return=True)
    async def gen() -> AsyncGenerator[int, str]:
        log.append('body:start')
        try:
            yield 1
            log.append('body:sleeping')
            await asyncio.sleep(3600)
            yield 2
        finally:
            log.append('body:finally')

    async def run():
        g = gen()
        assert await g.__anext__() == 1
        task = asyncio.create_task(g.__anext__())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.01)

    asyncio.run(run())
    assert log == ['body:start', 'body:sleeping', 'body:finally']


def test_async_iterator_is_read_only_passthrough():
    """`AsyncIterator[T]` has no send type: sent values are passed through per the protocol."""
    log: list[str] = []

    @validate_call(validate_return=True)
    async def gen() -> AsyncIterator[int]:
        cmd = yield 1
        log.append(f'cmd:{cmd!r}')
        yield 2

    async def run():
        g = gen()
        assert await g.__anext__() == 1
        # Anything sent is forwarded untouched (no send type to validate against):
        assert await g.asend(object()) == 2

    asyncio.run(run())
    assert len(log) == 1
    assert log[0].startswith('cmd:<object object at')


def test_async_gen_none_send_type_passthrough():
    log: list[str] = []

    @validate_call(validate_return=True)
    async def gen() -> AsyncGenerator[int, None]:
        cmd = yield 1
        log.append(f'cmd:{cmd!r}')
        yield 2

    async def run():
        g = gen()
        assert await g.__anext__() == 1
        assert await g.asend('unvalidated') == 2

    asyncio.run(run())
    assert log == ["cmd:'unvalidated'"]


def test_async_gen_first_send_must_be_none():
    log: list[str] = []
    traced = make_traced(log)

    async def run():
        gen = traced(1)
        with pytest.raises(TypeError, match="can't send non-None value"):
            await gen.asend('too-early')

    asyncio.run(run())
    assert log == []
