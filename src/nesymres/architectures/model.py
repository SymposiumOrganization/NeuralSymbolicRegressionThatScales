import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from .set_encoder import SetEncoder
#from .beam_search import *
from ..dclasses import Architecture


class Model(pl.LightningModule):
    def __init__(
        self,
        cfg: Architecture
    ):
        super().__init__()

        self.enc = SetEncoder(cfg)
        self.trg_pad_idx = cfg.trg_pad_idx
        self.tok_embedding = nn.Embedding(cfg.output_dim, cfg.dim_hidden)
        self.pos_embedding = nn.Embedding(cfg.length_eq, cfg.dim_hidden)
        if cfg.sinuisodal_embeddings:
            self.create_sinusoidal_embeddings(
                cfg.length_eq, cfg.dim_hidden, out=self.pos_embedding.weight
            )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dim_hidden,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dec_pf_dim,
            dropout=cfg.dropout,
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.dec_layers)
        self.fc_out = nn.Linear(cfg.dim_hidden, cfg.output_dim)
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dropout = nn.Dropout(cfg.dropout)
    

    def create_sinusoidal_embeddings(self, n_pos, dim, out):
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()
        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask, mask

    def forward(self,batch):
        b = batch[0].permute(0, 2, 1)
        size = b.shape[-1]
        src_x = b[:, :, : (size - 1)]
        src_y = b[:, :, -1].unsqueeze(2)
        trg = batch[1].long()
        trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])
        src_mask = None
        encoder_input = torch.cat((src_x, src_y), dim=-1)
        enc_src = self.enc(encoder_input) 
        assert not torch.isnan(enc_src).any()
        pos = self.pos_embedding(
            torch.arange(0, batch[1].shape[1] - 1)
            .unsqueeze(0)
            .repeat(batch[1].shape[0], 1)
            .type_as(trg)
        )
        te = self.tok_embedding(trg[:, :-1])
        #breakpoint()
        trg_ = self.dropout(te + pos)
        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            enc_src.permute(1, 0, 2),
            trg_mask2.bool(),
            tgt_key_padding_mask=trg_mask1.bool(),
        ) 
        output = self.fc_out(output)
        return output, trg

    def compute_loss(self,output, trg):
        output = output.permute(1, 0, 2).contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = self.criterion(output, trg)
        return loss

    def training_step(self, batch, _):
        output, trg = self.forward(batch)
        loss = self.compute_loss(output,trg)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        output, trg = self.forward(batch)
        loss = self.compute_loss(output,trg)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer


    def fitfunc(self, X,y):
        
        with torch.no_grad():
            mask = [([torch.sum(X[0,:,i] == 0) for i in range(X.shape[2])][j] == X.shape[1]).numpy() for j in range(X.shape[2])]
            x_bfgs = X.clone()
            for i in range(X.shape[2]):
                if mask[i] == True:
                    x_bfgs[:,:,i] = X[:,:,i]+1
            encoder_input = torch.cat((X, y), dim=-1)
            bfgs_input = torch.cat((x_bfgs, y), dim=-1)
            enc_src = self.enc(encoder_input)
            src_enc = enc_src
            bs = 1
            enc_src = (
                src_enc.unsqueeze(1)
                .expand((bs, beam_size) + src_enc.shape[1:])
                .contiguous()
                .view((bs * beam_size,) + src_enc.shape[1:])
            )
            print(
                "Memory footprint of the encoder: {}GB \n".format(
                    enc_src.element_size() * enc_src.nelement() / 10 ** (9)
                )
            )
            assert enc_src.size(0) == bs * beam_size
            generated = torch.zeros(
                [bs * beam_size, self.length_eq],
                dtype=torch.long,
                device=self.device,
            )
            generated[:, 0] = 1
            # trg_indexes = [[1] for i in range(bs*self.beam_size)]
            cache = {"slen": 0}
            # generated = torch.tensor(trg_indexes,device=self.device,dtype=torch.long)
            generated_hyps = [
                BeamHypotheses(beam_size, self.length_eq, 1.0, 1)
                for _ in range(bs)
            ]
            done = [False for _ in range(bs)]

            # Beam Scores
            beam_scores = torch.zeros(
                (bs, beam_size), device=self.device, dtype=torch.long
            )
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view(-1)

            cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
            while cur_len < self.length_eq:
                # breakpoint()
                generated_mask1, generated_mask2 = self.make_trg_mask(
                    generated[:, :cur_len]
                )

                # dec_args = (generated, enc_src, generated_mask, src_mask)

                pos = self.pos_embedding(
                    torch.arange(0, cur_len)  #### attention here
                    .unsqueeze(0)
                    .repeat(generated.shape[0], 1)
                    .type_as(generated)
                )
                te = self.tok_embedding(generated[:, :cur_len])
                trg_ = self.dropout(te + pos)

                output = self.dec(
                    trg_.permute(1, 0, 2),
                    enc_src.permute(1, 0, 2),
                    generated_mask2.float(),
                    tgt_key_padding_mask=generated_mask1.bool(),
                )
                output = self.fc_out(output)
                output = output.permute(1, 0, 2).contiguous()
                scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(
                    1
                )  # To Control
                assert output[:, -1:, :].shape == (
                    bs * beam_size,
                    1,
                    self.length_eq,
                )

                n_words = scores.shape[-1]
                # select next words with scores
                _scores = scores + beam_scores[:, None].expand_as(
                    scores
                )  # (bs * beam_size, n_words)
                _scores = _scores.view(
                    bs, beam_size * n_words
                )  # (bs, beam_size * n_words)

                next_scores, next_words = torch.topk(
                    _scores, 2 * beam_size, dim=1, largest=True, sorted=True
                )
                assert (
                    next_scores.size() == next_words.size() == (bs, 2 * beam_size)
                )

                # next batch beam content
                # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
                next_batch_beam = []

                # for each sentence
                for sent_id in range(bs):

                    # if we are done with this sentence
                    done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                        next_scores[sent_id].max().item()
                    )
                    if done[sent_id]:
                        next_batch_beam.extend(
                            [(0, self.trg_pad_idx, 0)] * beam_size
                        )  # pad the batch
                        continue

                    # next sentence beam content
                    next_sent_beam = []

                    # next words for this sentence
                    for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                        # get beam and word IDs
                        beam_id = idx // n_words
                        word_id = idx % n_words

                        # end of sentence, or next word
                        if (
                            word_id == self.env.word2id["F"]
                            or cur_len + 1 == self.length_eq
                        ):
                            generated_hyps[sent_id].add(
                                generated[
                                    sent_id * beam_size + beam_id,
                                    :cur_len,
                                ]
                                .clone()
                                .cpu(),
                                value.item(),
                            )
                        else:
                            next_sent_beam.append(
                                (value, word_id, sent_id * beam_size + beam_id)
                            )

                        # the beam for next step is full
                        if len(next_sent_beam) == beam_size:
                            break

                    # update next beam content
                    assert (
                        len(next_sent_beam) == 0
                        if cur_len + 1 == self.length_eq
                        else beam_size
                    )
                    if len(next_sent_beam) == 0:
                        next_sent_beam = [
                            (0, self.trg_pad_idx, 0)
                        ] * beam_size  # pad the batch
                    next_batch_beam.extend(next_sent_beam)
                    assert len(next_batch_beam) == beam_size * (sent_id + 1)

                # sanity check / prepare next batch
                assert len(next_batch_beam) == bs * beam_size
                # breakpoint()
                beam_scores = torch.tensor(
                    [x[0] for x in next_batch_beam], device=self.device
                )  # .type(torch.int64) Maybe #beam_scores.new_tensor([x[0] for x in next_batch_beam])
                beam_words = torch.tensor(
                    [x[1] for x in next_batch_beam], device=self.device
                )  # generated.new([x[1] for x in next_batch_beam])
                beam_idx = torch.tensor(
                    [x[2] for x in next_batch_beam], device=self.device
                )
                generated = generated[beam_idx, :]
                generated[:, cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen":
                        cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

                # update current length
                cur_len = cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )

            # def get_coeffs(s):
            #     roots = [int(s[i + 2]) for i, c in enumerate(s) if c == 'x']
            #     poly = np.poly1d(roots, r=True)
            #     coeffs = list(poly.coefficients.astype(np.int64))
            #     return [c % 10 for c in coeffs], coeffs

            # visualize hypotheses
            #print([len(x) for x in generated_hyps], cur_len)
            globals().update(locals())
            #!import code; code.interact(local=vars())
            supp1 = np.random.uniform(1, 3, [X.shape[2], 100])
            supp2 = np.ones([1, X.shape[1]])
            supp = np.concatenate([X[0].T.numpy(), supp2], axis=0)
            perc = 0
            cnt = 0
            gts = []
            best_preds = []
            best_preds_bfgs = []
            best_L = []
            best_L_bfgs = []
            for ii in tqdm(range(bs)):
                numbers_gt = y[ii]
                idx_nan_trg = np.where(np.isnan(numbers_gt))[0]
                idx_inf_trg = np.where(np.abs(numbers_gt) == np.inf)[0]
                idx_trg = np.unique(np.concatenate([idx_nan_trg, idx_inf_trg]))

                cnt = cnt + 1
                flag = 0
                L = []
                L_bfgs = []
                P = []
                P_bfgs = []
                counter = 1
                for ss, ww in sorted(
                    generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True
                ):
                    try:
                        ww = ww[1:].tolist()
                        ww = [x if x<14 else x+1 for x in ww]
                        pre = self.env._prefix_to_infix_benchmark(
                            self.env.de_tokenize(ww)
                        )
                        P.append(pre)
                        print('hypothesis #', counter, ': ',pre[0])
                        numbers_pred = lambdify("x,y,z,c", pre[0])(*supp)
                        idx_nan_pred = np.where(np.isnan(numbers_pred))[0]
                        idx_inf_pred = np.where(np.abs(numbers_pred) == np.inf)[0]
                        idx_pred = np.unique(
                            np.concatenate([idx_nan_pred, idx_inf_pred])
                        )
                        idx_nan = np.unique(np.concatenate([idx_pred, idx_trg]))
                        iii = np.setdiff1d(np.arange(0, X.shape[1]), idx_nan)
                        loss = np.mean(np.square(numbers_gt[iii].squeeze().numpy() - numbers_pred[iii]))
                        # print(loss)
                        L.append(np.abs(loss))
                        # try:

                        if bfgs_:
                            pred_w_c, constants, loss_bfgs, exa = bfgs(
                                pre[0], bfgs_input[ii], 10, self.env
                            )
                            L_bfgs.append(loss_bfgs)
                            P_bfgs.append(str(pred_w_c))

                        # except:
                        #    print('not yet implemented')
                        #    continue
                        # hh = " ".join(self.env.id2word[x] for x in ww.tolist())

                    except:
                        print("Math Syntax Error")
                        pass

                    counter = counter +1
                    # cc = get_coeffs(hh[4:])
                    # print(f"{ss:+.4f} {hh} || {cc[0]} || {cc[1]}")
                # if flag == 0:
                #     breakpoint()
                # print("")

                best_preds.append(P[np.nanargmin(L)][0])
                best_L.append(np.nanmin(L))
                if bfgs_:
                    best_preds_bfgs.append(P_bfgs[np.nanargmin(L_bfgs)])
                    best_L_bfgs.append(np.nanmin(L_bfgs))
                # FF.append(flag)
            # breakpoint()
            # select the best hypotheses
            tgt_len = torch.zeros(bs, device=self.device)  # src_len.new(bs)
            best = []

            for i, hypotheses in enumerate(generated_hyps):
                best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
                tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
                best.append(best_hyp)

            self.log(
                "score",
                perc * 100 / cnt,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "correct", perc, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )
            self.log(
                "tot", cnt, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

            if bfgs_:
                output = {'all_skel_outputs':P, 'all_skel_loss':L, 'all_bfgs_preds':P_bfgs, 'all_bfgs_loss':L_bfgs, 'best_skel_outputs':best_preds, 'best_skel_loss':best_L, 'best_bfgs_preds':best_preds_bfgs, 'best_bfgs_loss':best_L_bfgs}
                return output
            else:
                output = {'all_skel_outputs':P, 'all_skel_loss':L, 'best_skel_outputs':best_preds, 'best_skel_loss':best_L}
                return output



if __name__ == "__main__":
        model = SetTransformer(n_l_enc=2,src_pad_idx=0,trg_pad_idx=0,dim_input=6,output_dim=20,dim_hidden=40,dec_layers=1,num_heads=8,dec_pf_dim=40,dec_dropout=0,length_eq=30,lr=
            0.001,num_inds=20,ln=True,num_features=10,is_sin_emb=False, bit32=True,norm=False,activation='linear',linear=False,mean=torch.Tensor([1.]),std=torch.Tensor([1.]),input_normalization=False)
        src_x = torch.rand([2,5,20])
        src_y = torch.sin(torch.norm(src_x, dim=1)).unsqueeze(1)
        inp_1 = torch.cat([src_x,src_y], dim=1)
        inp_2 = torch.randint(0,13,[2,10])
        batch = (inp_1,inp_2)
        breakpoint()
        print(model)