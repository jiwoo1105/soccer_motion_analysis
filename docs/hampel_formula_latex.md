> **과거 방식 보관 문서.** 현재 어깨·골반 상대 정렬 지표와 1/3 총점은 [현재 계산법](current_evaluation.md)을 참고하세요. 아래 수식·주장은 당시 실험 설명이며 현재 적용 기준이 아닙니다.

# Hampel 이상치 판별 — 수식 정리 (노션 붙여넣기용)

> 노션에서 `/equation` (또는 `/수식`) 블록을 만들고 아래 LaTeX를 붙여넣으면 렌더링된다.
> 인라인은 `$$수식$$` 형태로 입력한다.

---

## 1. 기호 정의

```latex
\begin{aligned}
\theta_i &: \text{프레임 } i \text{의 회전각 (unwrap 완료, 단위 } ^\circ) \\
N &: \text{총 프레임 수}, \quad i = 0, 1, \dots, N-1 \\
k &: \text{윈도우 반폭 (half window)}, \quad k = 5 \\
\tau &: \text{판별 임계값}, \quad \tau = 10^\circ
\end{aligned}
```

---

## 2. 윈도우 정의

```latex
W_i = \left\{\, j \in \mathbb{Z} \;\middle|\; \max(0,\, i-k) \le j \le \min(N-1,\, i+k) \,\right\}
```

경계를 제외하면 $|W_i| = 2k+1 = 11$ 개의 프레임이다.

---

## 3. 기준값 (중앙값)

```latex
m_i = \operatorname{median}\left\{\, \theta_j \;\middle|\; j \in W_i \,\right\}
```

---

## 4. 판별식 ⭐ 핵심

```latex
b_i =
\begin{cases}
1, & \left|\theta_i - m_i\right| > \tau \quad (\text{이상치}) \\[4pt]
0, & \text{그 외} \quad (\text{정상})
\end{cases}
```

**한 줄로 압축한 형태:**

```latex
b_i = \mathbb{1}\!\left[\; \left|\,\theta_i - \operatorname{median}_{\,|j-i| \le k}\, \theta_j \,\right| > \tau \;\right]
```

---

## 5. 보간 (이상치 대체)

정상 프레임 집합을 $G = \{\, i \mid b_i = 0 \,\}$ 라 할 때, 이상치 프레임 $i$ 의 값은
양측 최근접 정상 프레임 사이의 선형 보간으로 대체한다.

```latex
a = \max\{\, j \in G \mid j < i \,\}, \qquad
c = \min\{\, j \in G \mid j > i \,\}
```

```latex
\tilde{\theta}_i =
\begin{cases}
\theta_i, & b_i = 0 \\[6pt]
\theta_a + \dfrac{i-a}{c-a}\left(\theta_c - \theta_a\right), & b_i = 1
\end{cases}
```

---

## 6. 네 가지 판별 방식 비교 (기준값 $r_i$ 만 다름)

공통 판별식:

```latex
b_i = \mathbb{1}\!\left[\, \left|\theta_i - r_i\right| > \tau \,\right]
```

기준값 $r_i$ 의 정의:

```latex
\begin{aligned}
\text{① old} \quad r_i &= \theta_{i-1} \\[6pt]
\text{② lastvalid} \quad r_i &= \theta_{a(i)}, \quad a(i) = \max\{\, j < i \mid b_j = 0 \,\} \\[6pt]
\text{③ avg} \quad r_i &= \frac{1}{n}\sum_{j=i-n}^{i-1} \theta_j, \quad n = 5 \\[6pt]
\text{④ Hampel} \quad r_i &= \operatorname{median}_{\,|j-i| \le k}\, \theta_j, \quad k = 5
\end{aligned}
```

**설계축 2×2 요약:**

```latex
\begin{array}{c|c|c}
 & \text{과거만 (단측)} & \text{전후 (중심)} \\ \hline
\text{평균 계열} & \text{② lastvalid, ③ avg} & - \\ \hline
\text{중앙값 계열} & - & \text{④ Hampel}
\end{array}
```

---

## 7. avg가 실패하는 이유 — 지연 편향

### 등속 근사

각속도 $\omega$ 로 회전 중일 때, 이전 $n$ 프레임 평균의 무게중심은 $t - \frac{n+1}{2}$ 이므로:

```latex
\theta_i - r_i^{\text{avg}} \;\approx\; \frac{n+1}{2}\,\omega \;\overset{n=5}{=}\; 3\omega
```

따라서 오탐 조건은:

```latex
3\omega > \tau \quad\Longleftrightarrow\quad \omega > \frac{\tau}{3} = 3.33\ ^\circ/\text{frame} = 100\ ^\circ/\text{s} \;\;(30\,\text{fps})
```

> **실측 검증**: 실제 영상 1298프레임에서
> $\operatorname{median}\!\left(\dfrac{|\theta_i - r_i^{\text{avg}}|}{3\omega_i}\right) = 0.99$

### 진동 신호에서의 정확한 형태

이전 $n$ 프레임 평균은 주파수 응답이 다음과 같은 선형 필터다:

```latex
H(\omega) = \frac{1}{n}\sum_{j=1}^{n} e^{-\mathrm{j}\omega j}
= \underbrace{\frac{\sin(n\omega/2)}{n\,\sin(\omega/2)}}_{D(\omega)\ \text{진폭 감쇠}}
\cdot
\underbrace{e^{-\mathrm{j}\omega\frac{n+1}{2}}}_{\text{위상 지연}}
```

진폭 $A$ 의 정현 신호에 대한 잔차 진폭:

```latex
R = A\left|\,1 - D(\omega)\,e^{-\mathrm{j}\,3\omega}\,\right|
```

> **주의**: 실제 영상의 지배 주기는 71~112프레임으로 길어 $D(\omega) \approx 0.99$ 이므로,
> 감쇠항은 무시 가능하고 등속 근사 $3\omega$ 가 더 정확하다 (실측 비율 0.99).

---

## 8. Hampel이 성립하는 이유

### (a) 단조 구간에서 잔차 = 0

$W_i$ 에서 $\theta$ 가 단조이면 정렬 순서가 인덱스 순서와 같으므로:

```latex
\operatorname{median}\left\{\theta_j\right\}_{j \in W_i} = \theta_i
\quad\Longrightarrow\quad
\left|\theta_i - m_i\right| = 0
```

**각속도와 무관하게** 잔차가 정확히 0이다.

### (b) 붕괴점 (breakdown point)

```latex
\varepsilon^{*}(\operatorname{median}) = \frac{1}{2}, \qquad
\varepsilon^{*}(\operatorname{mean}) = 0
```

윈도우 $|W_i| = 2k+1 = 11$ 에서 최대 $k = 5$ 개까지 오염되어도 $m_i$ 는 불변이다.
반면 평균은 단 하나의 이상치 $\theta_s$ 만으로 다음만큼 이동한다:

```latex
\Delta r^{\text{avg}} = \frac{\theta_s - \bar{\theta}}{n}
```

> 예: $n=5$, $\theta_s - \bar{\theta} = 55^\circ$ 이면 $\Delta r^{\text{avg}} = 11^\circ > \tau$

### (c) 한계 — 과반 오염

연속 오염 길이를 $L$ 이라 할 때:

```latex
L \le k \;\Rightarrow\; \text{전량 검출}, \qquad
L \ge k+1 \;\Rightarrow\; \text{중앙값이 오염값으로 이동하여 미검출}
```

$k=5$ 이므로 **5프레임까지 보장, 6프레임 이상은 놓친다.**

---

## 9. 전체 파이프라인

```latex
\begin{aligned}
&\text{(1)}\;\; \theta^{\text{raw}}_i = \operatorname{atan2}(v_{z,i},\, v_{x,i}) \\
&\text{(2)}\;\; \theta_i = \operatorname{unwrap}\left(\theta^{\text{raw}}_i\right) \\
&\text{(3)}\;\; b_i = \mathbb{1}\!\left[\left|\theta_i - \operatorname{median}_{|j-i|\le 5}\theta_j\right| > 10^\circ\right] \\
&\text{(4)}\;\; \tilde{\theta}_i = \text{선형 보간}\left(\theta,\, b\right) \\
&\text{(5)}\;\; \hat{\theta} = \mathrm{SG}_{41,\,2}\!\left(\tilde{\theta}\right) \\
&\text{(6)}\;\; \theta^{\text{det}} = \hat{\theta} - \mathrm{SG}_{81,\,2}\!\left(\hat{\theta}\right) \\
&\text{(7)}\;\; S_{\text{rot}} = \frac{1}{|E|-1}\sum_{p=1}^{|E|-1}\left|\theta^{\text{det}}_{e_{p+1}} - \theta^{\text{det}}_{e_p}\right|
\end{aligned}
```

여기서 $\mathrm{SG}_{w,\,d}$ 는 윈도우 $w$, 차수 $d$ 의 Savitzky–Golay 필터,
$E = \{e_1 < e_2 < \dots\}$ 는 `find_peaks(distance=10, prominence=3)` 로 검출한 극값 인덱스 집합이다.

> **순서 주의**: (3)이 (6)보다 앞서야 한다. 스파이크가 남은 상태로 baseline을 추정하면
> 주변 정상 구간까지 오염된다.

---

## 10. 논문용 주석 — 표준 Hampel과의 차이

문헌의 표준 Hampel identifier는 MAD로 정규화된 적응 임계값을 쓴다:

```latex
\left|\theta_i - m_i\right| > n_\sigma \cdot 1.4826 \cdot \operatorname{MAD}_i,
\qquad
\operatorname{MAD}_i = \operatorname{median}_{j \in W_i}\left|\theta_j - m_i\right|
```

본 연구는 **고정 절대 임계값** $\tau = 10^\circ$ 를 사용했다. 근거:

1. 판별 대상이 물리적으로 정의된 양(각도)이고, 사람 몸통이 낼 수 있는 각속도 상한이
   알려져 있어 절대 기준이 해석 가능하다.
2. MAD 방식은 구간마다 임계값이 변해 **드리블 사이클 간 판정 기준이 달라지는** 문제가 있다.
3. 비교 대상(old, lastvalid, avg)이 모두 고정 임계값이므로, **공정한 비교**를 위해 통일했다.

심사에서 지적될 수 있는 부분이므로 명시할 것.
