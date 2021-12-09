# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from dewloosh.geom import grid, PolyData
from dewloosh.geom.tri.trimesh import triangulate
from dewloosh.geom.topo.tr import Q4_to_T3
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import numpy as np
from copy import deepcopy

UZ, ROTX, ROTY, CX, CY, CXY, EXZ, EYZ, MX, MY, MXY, QX, QY = list(range(13))
labels = {UZ : 'UZ', ROTX : 'ROTX', ROTY : 'ROTY', CX : 'CX', 
          CY : 'CY', CXY : 'CXY', EXZ : 'EXZ', EYZ : 'EYZ', 
          MX : 'MX', MY : 'MY', MXY : 'MXY', QX : 'QX', QY : 'QY'}

model = 'mindlin'

size = Lx, Ly = (600., 800.)
E = 2890.
nu = 0.2
t = 50.

G = E/2/(1+nu)
D = np.array([[1, nu, 0], [nu, 1, 0], [0., 0, (1-nu)/2]]) * t**3 * (E / (1-nu**2)) / 12
S = np.array([[G, 0], [0, G]]) * t * 5 / 6

loads = {
    'LG1' : {
        'LC1' : {
            'type' : 'rect',
            'points' : [[0, 0], [Lx, Ly]],
            'value' : [0, 0, -0.01],
                },
        'LC2' : {
            'type' : 'rect',
            'region' : [0.2*Lx, 0.5*Ly, 0.2*Lx, 0.3*Ly],
            'value' : [0, 0, -0.1],
                }
            },
    'LG2' : {
        'LC3' : {
            'type' : 'point',
            'point' : [Lx/3, Ly/2],
            'value' : [0, 0, -10],
                },
        'LC4' : {
            'type' : 'point',
            'point' : [2*Lx/3, Ly/2],
            'value' : [0, 0, 10],
                }
            },
    'dummy1' : 10
        }



shape = nx, ny = (30, 40)
gridparams = {
    'size' : size,
    'shape' : shape,
    'origo' : (0, 0),
    'start' : 0,
    'eshape' : 'Q4'
    }
coords_, topo = grid(**gridparams)
coords = np.zeros((coords_.shape[0], 3))
coords[:, :2] = coords_[:, :]
del coords_
coords, triangles = Q4_to_T3(coords, topo)

triobj = triangulate(points=coords[:, :2], triangles=triangles)[-1]
Mesh = PolyData(coords=coords, topo=triangles)
centers = Mesh.centers()

from dewloosh.solid.navier import NavierProblem

Mindlin = NavierProblem(size, (50, 50), D=D, S=S, model=model)
LoadsM = Mindlin.add_loads_from_dict(deepcopy(loads))
Mindlin.solve()
Mindlin.postproc(centers[:, :2], cleanup=False)
res2dM = LoadsM['LG1', 'LC2'].res2d

aspects = {'x' : 1.0, 'y' : Ly/Lx, 'z' : 1.0}

fig = ff.create_trisurf(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                        simplices=triangles, color_func=res2dM[UZ, :],
                        colormap="Portland", title=labels[UZ], 
                        aspectratio=aspects, showbackground=True)


markdown_text = '''
### Dash and Markdown
A lot of text
'''

app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    dcc.Markdown(children=markdown_text),

    html.Div(children='''
        Dassssh: A web application framework for your data.
    '''),
          
    html.Label('Radio Items'),
    dcc.RadioItems(
        options=[
            {'label': 'Mindlin-Reissner', 'value': 'mindlin'},
            {'label': 'Kirchhoff-Love', 'value': 'kirchhoff'}
        ],
        value='mindlin'
    ),
    
     dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),
    
    dcc.Markdown(children="""
        # Material
        
        Enter material data in kN and cm.
                 
    """),
    
    html.Div([
        "E = ", dcc.Input(id='Young', value=E, type='number', placeholder="Young's modulus"),
        html.Div(),
        r"nu = ", dcc.Input(id='Poisson', value=nu, type='number', placeholder="Poisson's ratio"),
        html.Div(),
        "t = ", dcc.Input(id='Thickness', value=t, type='number', placeholder="thickness"),
        html.Div(id='my-div')
    ]),

    dcc.Graph(
        id='Navier solution',
        figure=fig
    )
])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='Young', component_property='value')]
)
def update_output_div(input_value):
    return 'You\'ve entered "{}"'.format(input_value)

if __name__ == '__main__':
    app.run_server(debug=True)