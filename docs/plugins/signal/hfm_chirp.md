---
SPDX-FileCopyrightText: openSTB contributors
SPDX-License-Identifier: BSD-2-Clause-Patent
---

# HFM chirp signal

**Plugin name**: `hfm_chirp`
<br>
**Implementation**: [`openstb.simulator.system.signal.HFMChirp`][]

A hyperbolic frequency modulated (LFM) chirp sweeps between two frequencies with a
hyperbolic rate of change. For a start frequency of $f_a$, an end frequency of $f_b$, a
chirp length of $\tau$ and an amplitude $A$, we can define a parameter

\[
b = \frac{1/f_b - 1/f_a}{\tau}.
\]

The signal is then

\[
s(t) = \begin{cases}
  A\exp\left(j2\pi\dfrac{\ln(1 + bf_at)}{b}\right) & 0 \leq t \leq \tau, \\
  0 & \text{otherwise}.
\end{cases}
\]

The instantaneous frequency of this is

\[
f(t) = \begin{cases}
  \dfrac{f_a}{1 + bf_at} & 0 \leq t \leq \tau, \\
  0 & \text{otherwise}.
\end{cases}
\]

which changes hyperbolically for the duration of the chirp as desired. The inverse of
this is the instantaneous period,

\[
T(t) = \begin{cases}
  \dfrac{1}{f_a} + bt & 0 \leq t \leq \tau, \\
  \infty & \text{otherwise}.
\end{cases}
\]

This is linear, giving rise to the alternate name of linear period modulated chirp. Note
that the frequency may increase ($f_b > f_a$) or decrease ($f_b < f_a$) during the
chirp.


## Parameters

Four parameters are required for the chirp:

* `f_start`: the start frequency ($f_a$) of the chirp in Hertz.
* `f_stop`: the stop frequency ($f_b$) of the chirp in Hertz.
* `duration`: the length of the chirp ($\tau$) in seconds.
* `rms_spl`: the root-mean-square sound pressure level of the transmitted chirp, i.e.,
  $20\log_{10} (A / 10^{-6})$.

For example, for an up-chirp sweeping a bandwidth of 20kHz over 10ms centred at 100kHz
with a SPL of 180dB (corresponding to 1000 Pascal):

```toml
[signal]
plugin = "hfm_chirp"
f_start = 90e3
f_stop = 110e3
duration = 10e-3
rms_spl = 180
```

Two optional parameters may also be given:

* `signal_window`: details of a window to be applied to the chirp, for example to taper
  the ends to reduce ringing. By default no window is applied.

* `rms_after_window`: if set to true (which is the default), the `rms_spl` parameter
  is applied to the signal after the signal window. If set to false, the `rms_spl`
  parameter is applied to the signal before the window.

For example, to apply a [Tukey window][] tapering the first and last 10% of the previous
signal:

```toml
[signal]
plugin = "hfm_chirp"
f_start = 90e3
f_stop = 110e3
duration = 10e-3
rms_spl = 180
rms_after_window = false

[signal.window]
plugin = "tukey"
alpha = 0.2
```

[Tukey window]: https://en.wikipedia.org/wiki/Window_function#Tukey_window

