import torch
import numpy as np

# * * * *  * * * *  * * * *   *       *   
# *        *     *  *     *   * *   * *   
# *   * *  * * *    * * * *   *   *   *   
# *     *  *     *  *     *   *       *   
# * * * *  *     *  *     *   *       *   

# THIS IS THE CORE PY CODE OF GRAM FRAMEWORK



def volume_computation2(language, other):

    """
    HyperAlign addition, in the exact style of the functions below: the 2-vector Gramian volume
    between language (shape [batch_size1, feature_dim]) and one other embedding
    (shape [batch_size2, feature_dim]), via the determinant of a 2x2 Gram matrix.

    For unit vectors det(G) = 1 - cos^2, so this volume is |sin(angle)| — the same geometry the
    main loss uses, at arity 2. Small volume = linearly dependent = aligned. Used by loss_doc so
    the whole model speaks one language (volume), never a separate cosine objective.

    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - other (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing the doc embedding h.

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]
    batch_size2 = other.shape[0]

    # Compute pairwise dot products for language with itself (shape: [batch_size1, batch_size2])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with the other embedding (shape: [batch_size1, batch_size2])
    lh = language @ other.T

    # Compute pairwise dot products for the other embedding with itself
    hh = torch.einsum('bi,bi->b', other, other).unsqueeze(0).expand(batch_size1, -1)

    # 2x2 determinant in closed form: det([[ll, lh], [lh, hh]]) = ll*hh - lh^2
    gram_det = (ll * hh - lh * lh).float()

    # Square root of the absolute determinant, exactly like the higher-arity functions.
    # eps guards the sqrt gradient at exactly 0
    res = torch.sqrt(torch.abs(gram_det) + 1e-8)
    return res


def volume_computation3(language, video, audio):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 3x3
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 3, 3])
    G = torch.stack([
        torch.stack([ll, lv, la], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa], dim=-1)  # Third row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def volume_computation4(language, video, audio, subtitles):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la, ls], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss], dim=-1)   # Fourth row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def volume_computation5(language, video, audio, subtitles, depth):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 5x5
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    - depth (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing depth features.    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T
    ld = language@depth.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    dd = torch.einsum('bi,bi->b', depth, depth).unsqueeze(0).expand(batch_size1, -1)
    dv = torch.einsum('bi,bi->b', depth, video).unsqueeze(0).expand(batch_size1, -1)
    da = torch.einsum('bi,bi->b', depth, audio).unsqueeze(0).expand(batch_size1, -1) 
    ds = torch.einsum('bi,bi->b', depth, subtitles).unsqueeze(0).expand(batch_size1, -1)


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 5, 5])
    G = torch.stack([
        torch.stack([ll, lv, la, ls, ld], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs, dv], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa, da], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss, ds], dim=-1),   # Fourth row of the Gram matrix
        torch.stack([ld, dv, da, ds, dd], dim=-1)
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def present_from_feats(feats, thr=0.5):
    """Per-clip modality-presence mask (B, L) from a list of L gallery-feature tensors (each (B, dim)).

    A modality is ABSENT for clip j when its feature is a (near-)zero vector -- the loader zero-fills
    any modality it could not load, and present features are L2-normed (norm==1). Used uniformly in
    train/eval/validation so a clip's volume is taken over exactly the modalities it actually has.
    """
    return torch.stack([(f.norm(dim=-1) > thr).float() for f in feats], dim=1)


def volume_computation_masked(language, inputs, present=None):
    """Arity-generic Gramian volume with per-clip missing-modality masking (HyperAlign addition).

    Same geometry as volume_computation, but each gallery clip may be MISSING some of the modalities
    the task asks for, and then its volume is taken over ONLY the modalities it actually has.

    language : (B1, dim) text/query features (text is always present -> never masked).
    inputs   : list of L tensors, each (B2, dim), the gallery modalities in a fixed order (e.g. V,A,S).
    present  : (B2, L) 0/1 (or bool). present[j, m] = 1 if gallery clip j really has modality m.
               A MISSING modality is turned into an orthonormal phantom axis: its Gram row/col becomes
               the identity row/col, so det(G) collapses to det(present sub-Gram) -- exactly the
               lower-arity volume for that clip (a missing axis contributes a factor 1, no noise).
               present=None (or all ones) reproduces volume_computation byte-for-byte (no regression).
               Built only from *, +, det -> fully differentiable, backprop-safe.

    Returns (B1, B2): the volume for each (language_i, gallery_j) pair, each at clip j's own arity.
    """
    B1 = language.shape[0]
    B2 = inputs[0].shape[0]
    L = len(inputs)
    M = L + 1                                             # language + L gallery modalities

    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, B2)
    l_inputs = [language @ x.T for x in inputs]           # language <-> each modality
    rows = [torch.stack([ll] + l_inputs, dim=-1)]         # first Gram row (language)
    for i in range(L):
        row = [l_inputs[i]]
        for j in range(L):
            row.append(torch.einsum('bi,bi->b', inputs[i], inputs[j]).unsqueeze(0).expand(B1, -1))
        rows.append(torch.stack(row, dim=-1))
    G = torch.stack(rows, dim=-2)                         # (B1, B2, M, M)

    if present is not None:
        present = present.to(G.dtype)                                            # (B2, L)
        ones = torch.ones(B2, 1, device=G.device, dtype=G.dtype)
        pres_full = torch.cat([ones, present], dim=1)                            # (B2, M) language always 1
        keep = (pres_full.unsqueeze(2) * pres_full.unsqueeze(1)).unsqueeze(0)    # (1, B2, M, M) zero missing row/col
        G = G * keep + torch.diag_embed(1.0 - pres_full).unsqueeze(0)           # phantom identity on missing diag

    gram_det = torch.det(G.float())
    res = torch.sqrt(torch.abs(gram_det) + 1e-8)
    return res


def volume_computation(language, *inputs):
    """
    General function to compute volume for contrastive learning loss functions.
    Compute the volume metric for each vector in language batch and all the other modalities listed in *inputs.

    Args:
    - language (torch.Tensor): Tensor of shape (batch_size1, dim)
    - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    batch_size1 = language.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Compute pairwise dot products for language with itself
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with each input
    l_inputs = [language @ input.T for input in inputs]

    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)

    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([ll] + l_inputs, dim=-1),
        *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
    ], dim=-2)

    # Compute the determinant for each Gram matrix
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    return res
