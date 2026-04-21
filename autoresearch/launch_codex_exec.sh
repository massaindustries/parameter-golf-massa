#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_TAG="${1:-mar24}"
BRANCH_NAME="autoresearch/${SESSION_TAG}"
PROMPT_FILE="$ROOT_DIR/autoresearch/_codex_prompt.txt"

if [ -f "$ROOT_DIR/autoresearch/runpod.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/autoresearch/runpod.env"
  set +a
fi

python3 "$ROOT_DIR/autoresearch/refresh_second_brain.py" >/dev/null 2>&1 || true

cat >"$PROMPT_FILE" <<EOF
Sei in /root/fmassapg sulla branch ${BRANCH_NAME}.

Obiettivo: eseguire un ciclo completo di autoresearch per la challenge Parameter Golf in autonomia, usando SOLO questa repo locale e il server corrente.

Prima di fare esperimenti:
- leggi autoresearch/program.md
- leggi autoresearch/second_brain_snapshot.md se esiste
- leggi autoresearch/second_brain.md se esiste
- leggi autoresearch/constraints.md
- leggi train_gpt.py
- conferma che i dati esistono in ./data/datasets/fineweb10B_sp1024/ e il tokenizer in ./data/tokenizers/
- usa autoresearch/results.tsv come registro dei risultati

Regole operative aggiuntive:
- lavora sempre dalla repo root /root/fmassapg
- NON introdurre network calls nei training run o nel codice di eval
- puoi usare il network solo per studiare il repository ufficiale openai/parameter-golf, la leaderboard e i record pubblici degli avversari
- NON modificare i file in autoresearch/ salvo:
  - autoresearch/results.tsv per registrare i risultati
  - autoresearch/program.md come parte del strategic review di fine ciclo
  - autoresearch/second_brain.md per il notebook sintetico
  - autoresearch/second_brain_snapshot.md tramite autoresearch/refresh_second_brain.py
- il cervello resta sempre su Seeweb: tu, gli eventuali coding agents, l'analisi, le mail, i merge e la strategia non devono mai spostarsi su Runpod
- NON usare local_autoresearch_queue.py come orchestratore: e deprecato e non soddisfa il workflow richiesto
- NON usare autoresearch/launch_runpod_worker.sh come cervello del loop: puo servire solo per un job remoto subordinato, mai come orchestrazione principale
- usa python3 autoresearch/dispatch_experiment.py per i run quando serve routing backend/fallback; usa bash autoresearch/run_experiment.sh solo per run esplicitamente locali o debug mirato
- usa il protocollo pieno: 20000 step, MAX_WALLCLOCK_SECONDS=0, nessun timeout operativo
- Seeweb rimane il server primario e la source of truth per branch promotion, results.tsv, mail e decisioni strategiche
- Runpod, se configurato, e solo spazio esecutivo remoto per training/eval: niente loop strategico remoto, niente controller remoto, niente agenti remoti
- non superare mai un run attivo su Seeweb; su Runpod puoi usare piu pod H100/H200 in parallelo, ma ogni pod deve eseguire un solo training run e il cervello deve restare qui
- su Runpod sii parsimonioso: scegli il miglior rapporto tempo/prezzo osservato, evita di lasciare pod costosi accesi senza probabilita realistica di completare il run, e usa autoresearch/report_runpod_value.py dopo i run remoti per confrontare H100/H200
- il saldo Runpod e limitato: con saldo basso non lanciare run remoti che non abbiano una chance realistica di completare un full 20000-step run; meglio restare solo su Seeweb che sprecare gli ultimi crediti
- prima di ogni run su Seeweb assicurati che la GPU locale sia completamente libera; se non lo e, aspetta e riprova
- se Runpod fallisce per provisioning, ssh, pod stoppato o credito insufficiente, smetti di mandarci run e torna su Seeweb senza chiedere approvazione
- quando una nuova idea richiede modifiche vere alla codebase, crea una candidate branch, testa lì, e fai merge nel ramo promosso solo se batte la codebase attuale
- se una candidate branch va peggio, non fare merge: lasciala come branch archivio dell'esperimento e torna al ramo promosso
- mantieni gli esperimenti atomici: una sola idea per commit
- la priorità di ricerca attuale è self-learning / TTT / backpropagation in inference post-training
- disciplina di ricerca obbligatoria:
  - prima di lanciare un run, esplicita nella tua reasoning interna cosa cambia rispetto al run precedente
  - non lanciare mai un run identico al precedente salvo rerun di conferma esplicito
  - non fare più di 3 full run consecutivi sulla stessa famiglia fallimentare
  - se 3 run consecutivi sulla stessa famiglia collassano, devi fare pivot di idea class prima del run successivo
  - se un adapting run è peggiore del suo anchor di oltre 0.20 bpb, considera quella famiglia strutturalmente rotta finché non c'è una modifica reale a `train_gpt.py`
- per ogni idea TTT devi verificare dai log che l'adattamento avvenga davvero e che migliori la metrica finale post-quant
- esegui prima la coda obbligatoria di rerun descritta in autoresearch/program.md
- poi esegui la self-learning buildout queue descritta in autoresearch/program.md
- dopo ogni run estrai val_bpb e size dai log prodotti in logs/
- il marker post-quant finale puo essere `final_int8_zlib_roundtrip_exact` oppure `final_int6_lzma_roundtrip_exact` a seconda del branch: parsare sempre il roundtrip finale realmente emesso dal `train_gpt.py` corrente, insieme alla corrispondente riga `Total submission size ...`
- dopo ogni ciclo aggiorna anche il second brain:
  - aggiorna autoresearch/second_brain.md in forma molto sintetica
  - esegui python3 autoresearch/refresh_second_brain.py
- dopo ogni run, dopo la riflessione locale e prima del test successivo, manda la mail di aggiornamento usando autoresearch/send_update_email.py senza chiedere approvazione
- dopo ogni singolo run devi fare micro-review, generare nuove proposte e includerle nella mail; "queue exhausted" da solo non e un esito accettabile
- non sostituire il loop strategico con un controller statico che esegue una queue finita e poi resta in idle
- conserva solo miglioramenti validi sotto 16000000 bytes
- se una modifica non migliora o viola il size limit, non promuoverla; gestiscila secondo le regole di branch/merge del programma
- completa esattamente un ciclo completo di ricerca per questa invocazione:
  - o un ciclo con un esperimento completo + parsing + results.tsv + micro-review + mail
  - oppure, se non e opportuno lanciare un run, un ciclo di analisi/strategic review che aggiorna piano e prossimi test
- poi termina pulitamente: un outer loop locale ti rilancera automaticamente per il ciclo successivo

Formato risultati:
- aggiorna autoresearch/results.tsv con le 7 colonne previste in program.md
- usa `status=promote` solo per miglioramenti/promozioni reali
- usa `status=anchor` per run di riferimento o no-update utili ma non migliori del best globale
- non usare `keep` come sinonimo ambiguo di "interessante"

Nota pratica:
- questa esecuzione stessa gira dentro tmux, quindi i processi del loop sono già persistenti rispetto alla sessione SSH
- se fai code review o riassunti finali, tienili brevi e poi termina: il loop persistente e gestito dal wrapper locale
EOF

exec codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --cd "$ROOT_DIR" \
  --skip-git-repo-check \
  - <"$PROMPT_FILE"
