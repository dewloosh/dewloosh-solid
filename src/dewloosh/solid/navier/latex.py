# -*- coding: utf-8 -*-
from pylatex import NoEscape


def expr_to_ltx(lhs, rhs, *args, env='{equation}', sign='=',
                dfrac=False, pre=None, post=None, **kwargs):
    if dfrac:
        lhs = lhs.replace('frac', 'dfrac')
        rhs = rhs.replace('frac', 'dfrac')
    if isinstance(pre, str):
        lhs = ' '.join([pre, lhs])
    if isinstance(post, str):
        rhs = ' '.join([rhs, post])
    return NoEscape(
        r"""
        \begin{env}
            {lhs} {sign} {rhs}
        \end{env}
        """.format(env = env,
                   lhs = lhs,
                   sign = sign,
                   rhs = rhs
                   )
        )


def expr_to_ltx_breqn(lhs, rhs, *args, env='{dmath}', **kwargs):
    return expr_to_ltx(lhs, rhs, *args, env=env, **kwargs)


def eq_to_ltx_multiline(lhs, rhs, *args, nsplit=2, **kwargs):
    kwargs['env'] = '{multline}'
