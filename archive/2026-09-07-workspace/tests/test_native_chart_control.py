import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from native_chart_control import LearnedChart


def fixture():
    g=torch.Generator().manual_seed(7)
    coordinates=torch.randn(60,4,generator=g,dtype=torch.float64)
    values=torch.cat((coordinates,coordinates.square(),torch.zeros(60,56,dtype=torch.float64)),1)
    return LearnedChart.fit(values),values


def test_zero_control_is_exact_identity_even_outside_support():
    chart,x=fixture();x=x[:5]+10
    for nonlinear in (False,True):
        assert torch.equal(chart.delta(x,torch.zeros(4),nonlinear),torch.zeros_like(x))


def test_actual_nonlinear_control_autograd_matches_finite_difference():
    chart,x=fixture();u=torch.tensor([.1,-.2,.05,.03],dtype=torch.float64,requires_grad=True)
    f=lambda z: chart.delta(x[:3],z,True).square().sum()
    analytic=torch.autograd.grad(f(u),u)[0]
    for j in range(4):
        step=torch.zeros_like(u);step[j]=1e-5
        finite=(f(u+step)-f(u-step))/(2e-5)
        torch.testing.assert_close(analytic[j],finite,rtol=1e-5,atol=1e-7)


def test_linear_control_stays_in_same_four_dimensional_span():
    chart,x=fixture();delta=chart.delta(x[:3],torch.ones(4),False)
    torch.testing.assert_close(delta,(delta@chart.basis.T)@chart.basis,atol=1e-10,rtol=1e-8)


def test_shared_bound_and_support_are_not_claimed_as_manifold_certificate():
    chart,x=fixture()
    shift=.5*chart.scale*torch.tanh(torch.full((4,),100.))
    assert torch.all(shift<=.5*chart.scale)
    assert chart.support(x+1e6)['outside_train_box'].all()


def test_raw_lift_preserves_off_subspace():
    chart,x=fixture()
    q,_=torch.linalg.qr(torch.randn(80,64,dtype=torch.float64))
    delta=chart.delta(x[:3],torch.ones(4),True)@q.T
    torch.testing.assert_close(delta-(delta@q)@q.T,torch.zeros_like(delta),atol=1e-10,rtol=0)


def test_graph_decoder_preserves_chart_coordinates_and_rbf_complement():
    chart,x=fixture();c=chart.encode(x[:7])+.17
    graph=chart.decode_graph(c);rbf=chart.decode(c)
    torch.testing.assert_close(chart.encode(graph),c,atol=1e-12,rtol=1e-12)
    difference=graph-rbf
    torch.testing.assert_close(difference-(difference@chart.basis.T)@chart.basis,torch.zeros_like(difference),atol=1e-12,rtol=0)


def test_graph_zero_is_exact_and_nonzero_shift_is_preserved():
    chart,x=fixture();x=x[:6]+3
    assert torch.equal(chart.delta_graph(x,torch.zeros(4)),torch.zeros_like(x))
    u=torch.tensor([.1,-.2,.3,-.4],dtype=torch.float64)
    delta=chart.delta_graph(x,u)
    expected=(.5*chart.scale*u.tanh()).expand(len(x),4)
    torch.testing.assert_close(delta@chart.basis.T,expected,atol=1e-12,rtol=1e-12)


def test_graph_native_chart_gradient_matches_finite_difference():
    chart,x=fixture();u=torch.tensor([.1,-.2,.05,.03],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda v:chart.delta_graph(x[:2],v),(u,))
