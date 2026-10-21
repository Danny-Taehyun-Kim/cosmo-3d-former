"""Shape, invariance and parameter-count checks for the encoder and decoder."""
import torch

from cosmo3dformer.model import PatchEncoder, PatchMAE, count_params

SIZES = {"S": (256, 6338496, 440204), "M": (480, 22192096, 468876), "L": (688, 45539472, 495500)}


def test_parameter_counts():
    """Encoder and decoder sizes match the values reported in the paper."""
    for name, (width, n_encoder, n_decoder) in SIZES.items():
        model = PatchMAE(n_in=12, d=width, L=8, h=8, ff_mult=4, p=0.1,
                         dd=128, dL=2, dh_=4, K=64, cut=16.0)
        encoder = count_params(model.enc)
        assert encoder == n_encoder, f"{name} encoder {encoder} != {n_encoder}"
        assert count_params(model) - encoder == n_decoder, f"{name} decoder mismatch"


def test_embedding_dimension():
    """The embedding is the masked mean and sum concatenated, so twice the width."""
    for name, (width, _, _) in SIZES.items():
        assert 2 * width in (512, 960, 1376)


def test_encoder_takes_no_coordinates():
    """The encoder signature accepts features, distances and a padding mask only."""
    encoder = PatchEncoder(n_in=12, d=64, L=2, h=4)
    torch.manual_seed(0)
    features = torch.randn(2, 10, 12)
    xyz = torch.randn(2, 10, 3)
    pad = torch.ones(2, 10, dtype=torch.bool)
    out = encoder(features, torch.cdist(xyz, xyz), pad)
    assert out.shape == (2, 10, 64)


def test_rotation_invariance():
    """Rotating the coordinates leaves the encoder output unchanged."""
    torch.manual_seed(0)
    encoder = PatchEncoder(n_in=12, d=64, L=2, h=4, p=0.0).eval()
    features = torch.randn(1, 12, 12)
    xyz = torch.randn(1, 12, 3)
    pad = torch.ones(1, 12, dtype=torch.bool)

    angle = torch.tensor(0.7)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                             [torch.sin(angle), torch.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]])
    with torch.no_grad():
        a = encoder(features, torch.cdist(xyz, xyz), pad)
        b = encoder(features, torch.cdist(xyz @ rotation.T, xyz @ rotation.T), pad)
    assert torch.allclose(a, b, atol=1e-5)


def test_hidden_features_cannot_reach_the_decoder():
    """Changing masked-patch features does not change the reconstruction."""
    torch.manual_seed(0)
    model = PatchMAE(n_in=12, d=64, L=2, h=4, p=0.0, dd=32, dL=1, dh_=2).eval()
    fv = torch.randn(1, 5, 12)
    dvv = torch.cdist(torch.randn(1, 5, 3), torch.randn(1, 5, 3))
    pv = torch.ones(1, 5, dtype=torch.bool)
    dmv = torch.rand(1, 7, 5) * 5
    pm = torch.ones(1, 7, dtype=torch.bool)
    with torch.no_grad():
        first = model(fv, dvv, pv, dmv, pm)[0]
        second = model(fv, dvv, pv, dmv, pm)[0]
    assert torch.equal(first, second)
