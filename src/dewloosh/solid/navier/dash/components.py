# -*- coding: utf-8 -*-
import plotly.figure_factory as ff
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html
import plotly.figure_factory as ff
import plotly.graph_objects as go


def fig3d(coords, triangles, res2d, cmap="Viridis", **params):
    Lx = params['Lx']
    Ly = params['Ly']
    aspects = {'x' : 1.0, 'y' : Ly/Lx, 'z' : 1.0}
    fig = ff.create_trisurf(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                            simplices=triangles, color_func=res2d,
                            colormap=cmap, title='', height=700,
                            aspectratio=aspects, showbackground=True)
    fig.update_layout(transition_duration=500,
                        scene=dict(
                            annotations=[
                                dict(
                                    showarrow=False,
                                    x=0.,
                                    y=0.,
                                    z=0.,
                                    text="A",
                                    xanchor="left",
                                    font=dict(
                                            color="black",
                                            size=20
                                        ),
                                    ),
                                dict(
                                    showarrow=False,
                                    x=Lx,
                                    y=0.,
                                    z=0.,
                                    text="B",
                                    xanchor="left",
                                    font=dict(
                                            color="black",
                                            size=20
                                        ),
                                    ),
                                dict(
                                    showarrow=False,
                                    x=Lx,
                                    y=Ly,
                                    z=0.,
                                    text="C",
                                    xanchor="left",
                                    font=dict(
                                            color="black",
                                            size=20
                                        ),
                                    ),
                                dict(
                                    showarrow=False,
                                    x=0.,
                                    y=Ly,
                                    z=0.,
                                    text="D",
                                    xanchor="left",
                                    font=dict(
                                            color="black",
                                            size=20
                                        ),
                                    ),
                                ]
                            )
                        )
    return fig


def fig2d(coords, res2d, **params):
    fig = go.Figure(data =
            go.Contour(
                x=coords[:, 0], 
                y=coords[:, 1],
                z=res2d,
            ))
    return fig


def input_mat(**params):
    E = params['E']
    nu = params['nu']
    return html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.InputGroupText("E"),
                    dbc.Input(
                        id='E',
                        placeholder="Young's modulus", 
                        type="number", 
                        value=E),
                    dbc.InputGroupText("kN/cm2"),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("nu"),
                    dbc.Input(
                        id='nu',
                        placeholder="Poisson's ratio", 
                        type="number", 
                        value=nu),
                    dbc.InputGroupText("-"),
                ],
                className="mb-3",
            ),
        ]
    )
    

def input_geom(**params):
    Lx = params['Lx']
    Ly = params['Ly']
    t = params['t']
    return html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Lx"),
                    dbc.Input(
                        id='Lx',
                        placeholder="Side length x", 
                        type="number", 
                        value=Lx),
                    dbc.InputGroupText("cm"),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Ly"),
                    dbc.Input(
                        id='Ly',
                        placeholder="Side length y", 
                        type="number", 
                        value=Ly),
                    dbc.InputGroupText("cm"),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("t"),
                    dbc.Input(
                        id='t',
                        placeholder="thickness", 
                        type="number", 
                        value=t),
                    dbc.InputGroupText("cm"),
                ],
                className="mb-3",
            ),
        ]
    )
    

def input_load(**params):
    x0 = params['x0']
    y0 = params['y0']
    w = params['w']
    h = params['h']
    q = params['q']
    return html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.InputGroupText("x0"),
                    dbc.Input(
                        id='x0',
                        placeholder="x coordinate of center", 
                        type="number", 
                        value=x0),
                    dbc.InputGroupText("cm"),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("y0"),
                    dbc.Input(
                        id='y0',
                        placeholder="y coordinate of center", 
                        type="number", 
                        value=y0),
                    dbc.InputGroupText("cm"),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("w"),
                    dbc.Input(
                        id='w',
                        placeholder="width", 
                        type="number", 
                        value=w),
                    dbc.InputGroupText("cm"),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("h"),
                    dbc.Input(
                        id='h',
                        placeholder="height", 
                        type="number", 
                        value=h),
                    dbc.InputGroupText("cm"),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("q"),
                    dbc.Input(
                        id='q',
                        placeholder="load intensity", 
                        type="number", 
                        value=-0.1),
                    dbc.InputGroupText("kN/cm2"),
                ],
                className="mb-3",
            ),
        ]
    )
    
    
def input_calc(**params):
    nx = params['nx']
    ny = params['ny']
    return html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Nx"),
                    dbc.Input(
                        id='nx',
                        type="number", 
                        value=nx),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Ny"),
                    dbc.Input(
                        id='ny',
                        type="number", 
                        value=ny),
                ],
                className="mb-3",
            ),
        ]
    )