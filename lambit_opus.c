// lambit_opus.c
// =============
// Fast C runtime for LamBit.
// Packed u32 heap, full deforestation, tail-call reuse.
// Compile: gcc -O3 -march=native -o lambit_opus lambit_opus.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

// Term Representation
// -------------------
// u16 pointer: tag(2 bits) | loc(14 bits)

#define TAG_NUL 0
#define TAG_BT0 1
#define TAG_BT1 2
#define TAG_TUP 3

#define MK(tag, loc) ((uint16_t)(((tag) << 14) | ((loc) & 0x3FFF)))
#define TAG(p)       ((p) >> 14)
#define LOC(p)       ((p) & 0x3FFF)

#define PTR_NUL MK(TAG_NUL, 1)
#define PTR_BT0 MK(TAG_BT0, 0)
#define PTR_BT1 MK(TAG_BT1, 0)

// Heap
// ----
// Packed u32: low 16 = fst, high 16 = snd.
// Single 32-bit load to read both halves.

#define HEAP_SIZE (1 << 14)
static uint32_t heap[HEAP_SIZE / 2] __attribute__((aligned(64)));
static uint16_t free_head = 0;

// Reads packed pair as u32 (fst in low 16, snd in high 16)
static inline uint32_t heap_pair(uint16_t loc) {
  return heap[loc >> 1];
}

// Extracts fst from packed pair
static inline uint16_t heap_fst(uint16_t loc) {
  return (uint16_t)heap[loc >> 1];
}

// Extracts snd from packed pair
static inline uint16_t heap_snd(uint16_t loc) {
  return (uint16_t)(heap[loc >> 1] >> 16);
}

// Writes packed pair
static inline void heap_set(uint16_t loc, uint16_t fst, uint16_t snd) {
  heap[loc >> 1] = ((uint32_t)snd << 16) | (uint32_t)fst;
}

// Allocates a 2-cell pair from the pre-seeded free list
static inline uint16_t alloc2(void) {
  uint16_t loc = free_head;
  free_head    = (uint16_t)heap[loc >> 1];
  return loc;
}

// Returns a pair to the free list
static inline void free_pair(uint16_t loc) {
  heap[loc >> 1] = (uint32_t)free_head;
  free_head      = loc;
}

// Allocates a tuple on the heap
static inline uint16_t make_tup(uint16_t fst, uint16_t snd) {
  uint16_t loc = alloc2();
  heap_set(loc, fst, snd);
  return MK(TAG_TUP, loc);
}

// Frees an unreferenced term tree (iterative on snd)
static void free_term(uint16_t p) {
  while (p >= 0xC000) {
    uint16_t loc = LOC(p);
    uint32_t pair = heap[loc >> 1];
    uint16_t f = (uint16_t)pair;
    uint16_t s = (uint16_t)(pair >> 16);
    free_pair(loc);
    if (f >= 0xC000) free_term(f);
    p = s;
  }
}

// Bytecodes
// ---------

#define OP_NUL      0x00
#define OP_BT0      0x01
#define OP_BT1      0x02
#define OP_LAM      0x03
#define OP_USE      0x04
#define OP_GET      0x05
#define OP_REC      0x06
#define OP_ERA      0x07
#define OP_VAR_BASE 0x08
#define OP_VAR_MAX  0x1F
#define OP_MAT_BASE 0x20
#define OP_MAT_MAX  0x8F
#define OP_TUP_BASE 0x90
#define OP_TUP_MAX  0xFF

#define MAX_CODE 8192
static uint8_t code[MAX_CODE];
static uint32_t code_len = 0;

// Stats
// -----

static uint64_t stat_fun = 0;
static uint64_t stat_lam = 0;
static uint64_t stat_mat = 0;
static uint64_t stat_get = 0;
static uint64_t stat_use = 0;

// Sub Stack
// ---------
// Var(k) = sub_stack[sub_sp - 1 - k]. No zeroing on read (affine).

#define MAX_SUB (1 << 12)
static uint16_t sub_stack[MAX_SUB] __attribute__((aligned(64)));
static uint32_t sub_sp = 0;

// Cont Stack
// ----------
// Packed u32: tag(16) | val(16).

#define CONT_TUP_SND  1
#define CONT_TUP_DONE 2
#define CONT_REC_FEED 3
#define CONT_REC_DONE 4

#define CONT_MAKE(tag, val) (((uint32_t)(tag) << 16) | ((uint32_t)(val) & 0xFFFF))
#define CONT_TAG(c)         ((c) >> 16)
#define CONT_VAL(c)         ((uint16_t)((c) & 0xFFFF))

#define MAX_CONT (1 << 12)
static uint32_t cont_stack[MAX_CONT] __attribute__((aligned(64)));
static uint32_t cont_sp = 0;

// Feed Buffer
// -----------
// Holds fst values for eval-feed fusion.

#define MAX_FEED_BUF 32
static uint16_t feed_buf[MAX_FEED_BUF];

// Precomputed fusibility: fusible[pc] = 1 iff REC at pc has fully-leaf argument
static uint8_t fusible[MAX_CODE];

// FeedSuper Table
// ---------------
// Precomputed feed dispatch: avoids per-opcode comparisons in feed_term_ctx.
// Packed uint16_t: low byte = ctrl, high byte = delta (MAT branch offset).
// Single 16-bit load per dispatch.

#define FEED_LAM     1
#define FEED_ERA     2
#define FEED_USE     3
#define FEED_MAT     4
#define FEED_GET     5
#define FEED_GET_MAT 6

#define FEED_PACK(ctrl, delta) ((uint16_t)((ctrl) | ((delta) << 8)))
#define FEED_CTRL(fs)          ((uint8_t)((fs) & 0xFF))
#define FEED_DELTA(fs)         ((uint8_t)((fs) >> 8))

static uint16_t feed_super[MAX_CODE];

// Bytecode Compiler
// -----------------

typedef struct { const char *src; int idx; int len; } Src;

static void src_skip(Src *s) {
  while (s->idx < s->len && (s->src[s->idx]==' ' || s->src[s->idx]=='\n' ||
         s->src[s->idx]=='\r' || s->src[s->idx]=='\t'))
    s->idx++;
}

static int src_match(Src *s, const char *str) {
  src_skip(s);
  int n = (int)strlen(str);
  if (s->idx + n <= s->len && memcmp(s->src + s->idx, str, n) == 0) {
    s->idx += n;
    return 1;
  }
  return 0;
}

static void src_expect(Src *s, const char *str) {
  if (!src_match(s, str)) {
    fprintf(stderr, "Expected '%s' at index %d\n", str, s->idx);
    exit(1);
  }
}

static char src_peek(Src *s) {
  src_skip(s);
  return s->idx < s->len ? s->src[s->idx] : '\0';
}

static int is_name_char(char c) {
  return (c>='a'&&c<='z') || (c>='A'&&c<='Z') || (c>='0'&&c<='9') || c=='_';
}

#define MAX_NAMES 64
static char cname_buf[MAX_NAMES][32];
static int  cname_used[MAX_NAMES];
static int  cname_depth = 0;

static void parse_name_into(Src *s, char *out, int maxlen) {
  src_skip(s);
  int start = s->idx;
  while (s->idx < s->len && is_name_char(s->src[s->idx])) s->idx++;
  int n = s->idx - start;
  if (n == 0) { fprintf(stderr, "Expected name at %d\n", s->idx); exit(1); }
  if (n >= maxlen) n = maxlen - 1;
  memcpy(out, s->src + start, n);
  out[n] = '\0';
}

// Finds name, returns de Bruijn INDEX (0 = innermost)
static int find_name_dbi(const char *nm) {
  for (int i = cname_depth - 1; i >= 0; i--)
    if (strcmp(cname_buf[i], nm) == 0)
      return (cname_depth - 1) - i;
  return -1;
}

static void emit(uint8_t b) {
  if (code_len >= MAX_CODE) { fprintf(stderr, "Code overflow\n"); exit(1); }
  code[code_len++] = b;
}

static void compile_func(Src *s);
static void compile_term(Src *s);

static int src_is_nul(Src *s) {
  src_skip(s);
  return s->idx+1 < s->len && s->src[s->idx]=='(' && s->src[s->idx+1]==')';
}

static int src_is_tup(Src *s) {
  src_skip(s);
  return s->idx < s->len && s->src[s->idx]=='(' &&
         (s->idx+1 >= s->len || s->src[s->idx+1]!=')');
}

// Matches UTF-8 λ followed by !
static int match_lambda_bang(Src *s) {
  src_skip(s);
  if (s->idx + 2 < s->len &&
      (uint8_t)s->src[s->idx] == 0xCE &&
      (uint8_t)s->src[s->idx+1] == 0xBB &&
      s->src[s->idx+2] == '!') {
    s->idx += 3;
    return 1;
  }
  return 0;
}

// Matches UTF-8 λ
static int match_lambda(Src *s) {
  src_skip(s);
  if (s->idx + 1 < s->len &&
      (uint8_t)s->src[s->idx] == 0xCE &&
      (uint8_t)s->src[s->idx+1] == 0xBB) {
    s->idx += 2;
    return 1;
  }
  return 0;
}

// Compiles a function (lambda/mat/use/get/term)
static void compile_func(Src *s) {
  src_skip(s);
  if (match_lambda_bang(s)) {
    emit(OP_GET);
    compile_func(s);
    return;
  }
  if (match_lambda(s)) {
    if (src_match(s, "{")) {
      src_expect(s, "0"); src_expect(s, ":");
      uint32_t mat_pos = code_len;
      emit(0);
      uint32_t z_start = code_len;
      compile_func(s);
      uint32_t z_end = code_len;
      src_match(s, ";");
      src_expect(s, "1"); src_expect(s, ":");
      compile_func(s);
      src_match(s, ";");
      src_expect(s, "}");
      uint32_t d = z_end - z_start;
      code[mat_pos] = (uint8_t)(OP_MAT_BASE + d);
      return;
    }
    if (src_match(s, "()")) {
      src_expect(s, ".");
      emit(OP_USE);
      compile_func(s);
      return;
    }
    char nm[32];
    parse_name_into(s, nm, 32);
    src_expect(s, ".");
    uint32_t lam_pos = code_len;
    emit(0);
    strcpy(cname_buf[cname_depth], nm);
    cname_used[cname_depth] = 0;
    cname_depth++;
    compile_func(s);
    cname_depth--;
    code[lam_pos] = cname_used[cname_depth] ? OP_LAM : OP_ERA;
    return;
  }
  compile_term(s);
}

// Compiles a term (rec/nul/tup/bit/var)
static void compile_term(Src *s) {
  src_skip(s);
  if (src_match(s, "~")) { emit(OP_REC); compile_term(s); return; }
  if (src_is_nul(s)) { src_match(s, "()"); emit(OP_NUL); return; }
  if (src_is_tup(s)) {
    src_match(s, "(");
    uint32_t tup_pos = code_len;
    emit(0);
    uint32_t f_start = code_len;
    compile_term(s);
    uint32_t f_end = code_len;
    src_expect(s, ",");
    compile_term(s);
    src_expect(s, ")");
    code[tup_pos] = (uint8_t)(OP_TUP_BASE + (f_end - f_start));
    return;
  }
  char ch = src_peek(s);
  if (ch == '0' && (s->idx+1 >= s->len || !is_name_char(s->src[s->idx+1]))) {
    s->idx++; emit(OP_BT0); return;
  }
  if (ch == '1' && (s->idx+1 >= s->len || !is_name_char(s->src[s->idx+1]))) {
    s->idx++; emit(OP_BT1); return;
  }
  char nm[32];
  parse_name_into(s, nm, 32);
  int dbi = find_name_dbi(nm);
  if (dbi < 0) { fprintf(stderr, "Unbound variable '%s'\n", nm); exit(1); }
  cname_used[cname_depth - 1 - dbi] = 1;
  emit((uint8_t)(OP_VAR_BASE + dbi));
}

// Disassembler
// ------------

static void print_code(uint32_t pc, int indent) {
  uint8_t op = code[pc];
  if (op == OP_NUL) { printf("NUL"); return; }
  if (op == OP_BT0) { printf("BT0"); return; }
  if (op == OP_BT1) { printf("BT1"); return; }
  if (op >= OP_VAR_BASE && op <= OP_VAR_MAX) {
    printf("VAR(%d)", op - OP_VAR_BASE); return;
  }
  if (op == OP_LAM) { printf("LAM "); print_code(pc+1, indent); return; }
  if (op == OP_ERA) { printf("ERA "); print_code(pc+1, indent); return; }
  if (op == OP_USE) { printf("USE "); print_code(pc+1, indent); return; }
  if (op == OP_GET) { printf("GET "); print_code(pc+1, indent); return; }
  if (op == OP_REC) { printf("REC "); print_code(pc+1, indent); return; }
  if (op >= OP_MAT_BASE && op <= OP_MAT_MAX) {
    uint32_t d = op - OP_MAT_BASE;
    printf("MAT(d=%u) {0: ", d);
    print_code(pc+1, indent+2);
    printf("; 1: ");
    print_code(pc+1+d, indent+2);
    printf("}");
    return;
  }
  if (op >= OP_TUP_BASE) {
    uint32_t d = op - OP_TUP_BASE;
    printf("TUP(d=%u) (", d);
    print_code(pc+1, indent+2);
    printf(", ");
    print_code(pc+1+d, indent+2);
    printf(")");
    return;
  }
  printf("?0x%02x", op);
}

// Fusibility Check
// -----------------
// A REC argument is fusible if it's a flat TUP(d=1) chain with all leaf values.

static inline int is_fusible_arg(const uint8_t *cc, uint32_t pc) {
  while (cc[pc] == OP_TUP_BASE + 1) {
    uint8_t f = cc[pc + 1];
    if (!(f <= OP_VAR_MAX && f != OP_LAM && f != OP_USE && f != OP_GET &&
          f != OP_REC && f != OP_ERA))
      return 0;
    pc += 2;
  }
  uint8_t s = cc[pc];
  return s == OP_BT0 || s == OP_BT1 || s == OP_NUL ||
         (s >= OP_VAR_BASE && s <= OP_VAR_MAX);
}

// Feed
// ----
// Feeds a heap term into the program func at pc.
// Pushes LAM bindings onto sub_stack. Returns pc of resulting body.

// Stats context — passed to feed so eval can keep counters in locals
typedef struct { uint64_t fun, lam, mat, get, use; } Stats;

// Fully iterative feed with stats context (uses precomputed FeedSuper table)
static __attribute__((hot)) uint32_t
feed_term_ctx(uint32_t pc, uint16_t term, Stats *restrict st) {
  uint16_t fstk[32];
  uint32_t fsp   = 0;
  uint16_t fh    = free_head;
  uint32_t sp    = sub_sp;
  const uint16_t *fs = feed_super;
  for (;;) {
    uint16_t entry = fs[pc];
    uint8_t  ctrl  = FEED_CTRL(entry);
    if (__builtin_expect(ctrl == FEED_GET_MAT, 1)) {
      st->get++;
      st->mat++;
      uint16_t loc  = LOC(term);
      uint32_t pair = heap[loc >> 1];
      uint16_t fst  = (uint16_t)pair;
      uint16_t snd  = (uint16_t)(pair >> 16);
      heap[loc >> 1] = (uint32_t)fh;
      fh = loc;
      uint32_t d = FEED_DELTA(entry);
      pc   = (fst == PTR_BT0) ? pc + 2 : pc + 2 + d;
      term = snd;
      continue;
    }
    if (ctrl == FEED_GET) {
      st->get++;
      uint16_t loc  = LOC(term);
      uint32_t pair = heap[loc >> 1];
      uint16_t fst  = (uint16_t)pair;
      uint16_t snd  = (uint16_t)(pair >> 16);
      heap[loc >> 1] = (uint32_t)fh;
      fh = loc;
      fstk[fsp++] = snd;
      pc  += 1;
      term = fst;
      continue;
    }
    // Writeback free_head and sub_sp for non-GET paths
    free_head = fh;
    sub_sp    = sp;
    if (ctrl == FEED_LAM) {
      st->lam++;
      sub_stack[sp++] = term;
      sub_sp = sp;
      pc += 1;
      if (fsp == 0) return pc;
      term = fstk[--fsp];
      continue;
    }
    if (ctrl == FEED_ERA) {
      st->lam++;
      free_term(term);
      fh = free_head;
      sub_stack[sp++] = 0;
      sub_sp = sp;
      pc += 1;
      if (fsp == 0) return pc;
      term = fstk[--fsp];
      continue;
    }
    if (ctrl == FEED_MAT) {
      st->mat++;
      uint32_t d = FEED_DELTA(entry);
      pc = (term == PTR_BT0) ? pc + 1 : pc + 1 + d;
      if (fsp == 0) return pc;
      term = fstk[--fsp];
      continue;
    }
    // ctrl == FEED_USE
    st->use++;
    pc += 1;
    if (fsp == 0) return pc;
    term = fstk[--fsp];
  }
}

// Wrapper for initial feed (before eval)
static uint32_t feed_term(uint32_t pc, uint16_t term) {
  Stats st = { stat_fun, stat_lam, stat_mat, stat_get, stat_use };
  uint32_t r = feed_term_ctx(pc, term, &st);
  stat_fun = st.fun; stat_lam = st.lam; stat_mat = st.mat;
  stat_get = st.get; stat_use = st.use;
  return r;
}

// Rebuilds a heap tuple from buf[bi..len-1] + snd (bottom-up)
static uint16_t rebuild_from_buf(uint16_t *buf, uint32_t bi,
                                 uint32_t len, uint16_t snd) {
  uint16_t val = snd;
  for (int i = (int)len - 1; i >= (int)bi; i--)
    val = make_tup(buf[i], val);
  return val;
}

// Feeds from buffer instead of heap (no alloc, no free for dispatch)
static uint32_t feed_from_buf_ctx(uint32_t pc, uint16_t *buf,
                                  uint32_t bi, uint32_t len,
                                  uint16_t snd, Stats *restrict st) {
  const uint16_t *fs = feed_super;
  for (;;) {
    if (bi >= len)
      return feed_term_ctx(pc, snd, st);
    uint16_t entry = fs[pc];
    uint8_t  ctrl  = FEED_CTRL(entry);
    if (ctrl == FEED_GET_MAT) {
      st->get++;
      st->mat++;
      uint16_t fst = buf[bi++];
      uint32_t d   = FEED_DELTA(entry);
      pc = (fst == PTR_BT0) ? pc + 2 : pc + 2 + d;
      continue;
    }
    if (ctrl == FEED_GET) {
      st->get++;
      uint16_t fst = buf[bi++];
      pc = feed_term_ctx(pc + 1, fst, st);
      continue;
    }
    if (ctrl == FEED_LAM) {
      st->lam++;
      sub_stack[sub_sp++] = rebuild_from_buf(buf, bi, len, snd);
      return pc + 1;
    }
    if (ctrl == FEED_ERA) {
      st->lam++;
      for (uint32_t i = bi; i < len; i++) free_term(buf[i]);
      free_term(snd);
      sub_stack[sub_sp++] = 0;
      return pc + 1;
    }
    if (ctrl == FEED_USE) {
      st->use++;
      return pc + 1;
    }
    return feed_term_ctx(pc, rebuild_from_buf(buf, bi, len, snd), st);
  }
}

// Evaluator
// ---------
// Goto dispatch, packed conts, tail-call reuse, full deforestation.

// Precomputed: program-level MAT deltas for inline GET+MAT
static uint32_t prog_mat_delta  = 0;
static uint32_t prog_mat_delta0 = 0;  // second MAT delta for branch 0
static uint32_t prog_mat_delta1 = 0;  // second MAT delta for branch 1

static uint16_t __attribute__((hot)) eval(uint32_t start_pc, uint32_t csp) {
  uint32_t pc  = start_pc;
  uint16_t val = 0;
  uint32_t saved_sp;
  const uint8_t *cc = code;
  Stats st = { stat_fun, stat_lam, stat_mat, stat_get, stat_use };

  // ---- EVAL: dispatch on opcode ----
  eval: {
    uint8_t op = cc[pc];

    // Most common: TUP (opcode >= 0x90)
    if (__builtin_expect(op >= OP_TUP_BASE, 1)) {
      uint32_t d = op - OP_TUP_BASE;
      if (__builtin_expect(d == 1, 1)) {
        uint8_t fst_op = cc[pc + 1];
        uint16_t fst_val;
        if      (fst_op == OP_BT0) { fst_val = PTR_BT0; }
        else if (fst_op == OP_BT1) { fst_val = PTR_BT1; }
        else if (fst_op == OP_NUL) { fst_val = PTR_NUL; }
        else if (fst_op >= OP_VAR_BASE && fst_op <= OP_VAR_MAX) {
          fst_val = sub_stack[sub_sp - 1 - (fst_op - OP_VAR_BASE)];
        } else {
          goto tup_slow;
        }
        cont_stack[csp++] = CONT_MAKE(CONT_TUP_DONE, fst_val);
        pc += 2;
        goto eval;
      }
      tup_slow:
      cont_stack[csp++] = CONT_MAKE(CONT_TUP_SND, pc + 1 + d);
      pc += 1;
      goto eval;
    }

    if (op == OP_REC) {
      st.fun++;
      // Tail-call reuse: if top is REC_DONE, convert to REC_FEED
      if (csp > 0 && CONT_TAG(cont_stack[csp - 1]) == CONT_REC_DONE) {
        saved_sp = CONT_VAL(cont_stack[csp - 1]);
        cont_stack[csp - 1] = CONT_MAKE(CONT_REC_FEED, saved_sp);
      } else {
        saved_sp = sub_sp;
        cont_stack[csp++] = CONT_MAKE(CONT_REC_FEED, sub_sp);
      }
      // Eval-feed fusion: if arg is fusible, collect directly
      if (fusible[pc]) {
        pc += 1;
        goto eval_rec_arg;
      }
      pc += 1;
      goto eval;
    }
    if (op == OP_NUL) { val = PTR_NUL; goto ret; }
    if (op == OP_BT0) { val = PTR_BT0; goto ret; }
    if (op == OP_BT1) { val = PTR_BT1; goto ret; }
    if (op >= OP_VAR_BASE) {
      val = sub_stack[sub_sp - 1 - (op - OP_VAR_BASE)];
      goto ret;
    }
    __builtin_unreachable();
  }

  // ---- RET: process continuation ----
  ret: {
    if (__builtin_expect(csp == 0, 0)) {
      stat_fun = st.fun; stat_lam = st.lam; stat_mat = st.mat;
      stat_get = st.get; stat_use = st.use;
      return val;
    }
    uint32_t c = cont_stack[csp - 1];

    // Most common: TUP_DONE chain.
    // Collect fst values into local array INSTEAD of make_tup.
    // If chain ends at REC_FEED: feed directly (full deforestation).
    // If chain ends elsewhere: build tuples from collected values.
    if (__builtin_expect(CONT_TAG(c) == CONT_TUP_DONE, 1)) {
      uint16_t fsts[32];
      uint32_t nf = 0;
      for (;;) {
        fsts[nf++] = CONT_VAL(c);
        csp--;
        if (__builtin_expect(csp > 0, 1)) {
          c = cont_stack[csp - 1];
          if (__builtin_expect(CONT_TAG(c) == CONT_TUP_DONE, 1))
            continue;
          // Full deforestation: chain ended at REC_FEED
          if (CONT_TAG(c) == CONT_REC_FEED) {
            saved_sp = CONT_VAL(c);
            sub_sp   = saved_sp;
            cont_stack[csp - 1] = CONT_MAKE(CONT_REC_DONE, saved_sp);
            // fsts are in reverse order: [0]=innermost, [nf-1]=outermost
            // Inline first GET+MAT with outermost fst
            st.get++;
            st.mat++;
            uint16_t fst0 = fsts[nf - 1];
            uint32_t npc  = (fst0 == PTR_BT0) ? 2 : 2 + prog_mat_delta;
            // Inline second GET+MAT with next-outermost fst
            uint16_t entry_d = feed_super[npc];
            uint8_t  ctrl_d  = FEED_CTRL(entry_d);
            if (__builtin_expect(ctrl_d == FEED_GET_MAT && nf > 1, 1)) {
              st.get++;
              st.mat++;
              uint16_t fst1 = fsts[nf - 2];
              uint32_t dd   = FEED_DELTA(entry_d);
              npc = (fst1 == PTR_BT0) ? npc + 2 : npc + 2 + dd;
              nf -= 2;
            } else {
              nf -= 1;
            }
            // Feed remaining fsts (reversed) + val as snd
            for (uint32_t i = 0; i < nf; i++)
              feed_buf[i] = fsts[nf - 1 - i];
            uint32_t body_pc = feed_from_buf_ctx(npc, feed_buf, 0, nf, val, &st);
            if (cc[body_pc] == OP_REC && fusible[body_pc]) {
              st.fun++;
              pc = body_pc + 1;
              goto eval_rec_arg;
            }
            if (cc[body_pc] == OP_REC) {
              st.fun++;
              cont_stack[csp - 1] = CONT_MAKE(CONT_REC_FEED, saved_sp);
              pc = body_pc + 1;
            } else {
              pc = body_pc;
            }
            goto eval;
          }
        }
        break;
      }
      // Fallback: build tuples from collected fsts
      for (uint32_t i = 0; i < nf; i++)
        val = make_tup(fsts[i], val);
      goto ret;
    }

    // TUP_SND: fst evaluated, now eval snd
    if (CONT_TAG(c) == CONT_TUP_SND) {
      cont_stack[csp - 1] = CONT_MAKE(CONT_TUP_DONE, val);
      pc = CONT_VAL(c);
      goto eval;
    }

    // REC_FEED: argument evaluated, feed into program
    if (CONT_TAG(c) == CONT_REC_FEED) {
      saved_sp = CONT_VAL(c);
      sub_sp   = saved_sp;
      cont_stack[csp - 1] = CONT_MAKE(CONT_REC_DONE, saved_sp);
      goto post_feed;
    }

    // REC_DONE: body returned, restore sub_sp
    if (CONT_TAG(c) == CONT_REC_DONE) {
      sub_sp = CONT_VAL(c);
      csp--;
      goto ret;
    }

    __builtin_unreachable();
  }

  // ---- POST_FEED: inline two levels of GET+MAT, then feed_term ----
  post_feed: {
    st.get++;
    uint16_t fh    = free_head;
    uint16_t loc0  = LOC(val);
    uint32_t pair0 = heap[loc0 >> 1];
    uint16_t fst0  = (uint16_t)pair0;
    uint16_t snd0  = (uint16_t)(pair0 >> 16);
    heap[loc0 >> 1] = (uint32_t)fh;
    fh = loc0;
    st.mat++;
    uint32_t npc = (fst0 == PTR_BT0) ? 2 : 2 + prog_mat_delta;
    // Second level: inline GET+MAT using precomputed deltas
    {
      st.get++;
      st.mat++;
      uint16_t loc1  = LOC(snd0);
      uint32_t pair1 = heap[loc1 >> 1];
      uint16_t fst1  = (uint16_t)pair1;
      uint16_t snd1  = (uint16_t)(pair1 >> 16);
      heap[loc1 >> 1] = (uint32_t)fh;
      fh = loc1;
      uint32_t d1  = (fst0 == PTR_BT0) ? prog_mat_delta0 : prog_mat_delta1;
      uint32_t npc2 = (fst1 == PTR_BT0) ? npc + 2 : npc + 2 + d1;
      free_head = fh;
      uint32_t body_pc = feed_term_ctx(npc2, snd1, &st);
      if (cc[body_pc] == OP_REC && fusible[body_pc]) {
        st.fun++;
        pc = body_pc + 1;
        goto eval_rec_arg;
      }
      if (cc[body_pc] == OP_REC) {
        st.fun++;
        cont_stack[csp - 1] = CONT_MAKE(CONT_REC_FEED, saved_sp);
        pc = body_pc + 1;
      } else {
        pc = body_pc;
      }
      goto eval;
    }
  }

  // ---- EVAL_REC_ARG: collect fusible TUP(d=1) chain, feed directly ----
  eval_rec_arg: {
    uint32_t feed_bi = 0;

    eval_rec_arg_collect: {
      uint8_t op = cc[pc];
      if (op == OP_TUP_BASE + 1) {
        uint8_t fst_op = cc[pc + 1];
        uint16_t fst_val;
        if      (fst_op == OP_BT0) { fst_val = PTR_BT0; }
        else if (fst_op == OP_BT1) { fst_val = PTR_BT1; }
        else if (fst_op == OP_NUL) { fst_val = PTR_NUL; }
        else if (fst_op >= OP_VAR_BASE && fst_op <= OP_VAR_MAX) {
          fst_val = sub_stack[sub_sp - 1 - (fst_op - OP_VAR_BASE)];
        } else {
          goto eval_rec_arg_fallback;
        }
        feed_buf[feed_bi++] = fst_val;
        pc += 2;
        goto eval_rec_arg_collect;
      }
      // Snd: must be leaf
      uint16_t snd_val;
      if      (op == OP_BT0) { snd_val = PTR_BT0; }
      else if (op == OP_BT1) { snd_val = PTR_BT1; }
      else if (op == OP_NUL) { snd_val = PTR_NUL; }
      else if (op >= OP_VAR_BASE && op <= OP_VAR_MAX) {
        snd_val = sub_stack[sub_sp - 1 - (op - OP_VAR_BASE)];
      } else {
        goto eval_rec_arg_fallback;
      }
      // Feed buffer + snd into program (inline two GET+MAT levels)
      {
        sub_sp = saved_sp;
        cont_stack[csp - 1] = CONT_MAKE(CONT_REC_DONE, saved_sp);
        st.get++;
        st.mat++;
        uint16_t ifst = feed_buf[0];
        uint32_t inpc = (ifst == PTR_BT0) ? 2 : 2 + prog_mat_delta;
        // Inline second GET+MAT if available
        uint32_t body_pc;
        uint16_t entry2 = feed_super[inpc];
        uint8_t  ctrl2  = FEED_CTRL(entry2);
        if (__builtin_expect(ctrl2 == FEED_GET_MAT && feed_bi > 1, 1)) {
          st.get++;
          st.mat++;
          uint16_t ifst2 = feed_buf[1];
          uint32_t d2    = FEED_DELTA(entry2);
          uint32_t inpc2 = (ifst2 == PTR_BT0) ? inpc + 2 : inpc + 2 + d2;
          body_pc = feed_from_buf_ctx(inpc2, feed_buf, 2, feed_bi, snd_val, &st);
        } else {
          body_pc = feed_from_buf_ctx(inpc, feed_buf, 1, feed_bi, snd_val, &st);
        }
        if (cc[body_pc] == OP_REC) {
          st.fun++;
          if (fusible[body_pc]) {
            // Tight tail-call: compact new bindings, loop
            uint32_t new_count = sub_sp - saved_sp;
            pc      = body_pc + 1;
            feed_bi = 0;
            goto eval_rec_arg_collect;
          }
          cont_stack[csp - 1] = CONT_MAKE(CONT_REC_FEED, saved_sp);
          pc = body_pc + 1;
          goto eval;
        }
        pc = body_pc;
        goto eval;
      }
    }

    eval_rec_arg_fallback: {
      // Not fusible — push remaining as CONT_TUP_DONEs, resume normal eval
      for (uint32_t i = 0; i < feed_bi; i++)
        cont_stack[csp++] = CONT_MAKE(CONT_TUP_DONE, feed_buf[i]);
      goto eval;
    }
  }
}

// Printer
// -------

// Prints a term to stdout
static void print_term(uint16_t p) {
  switch (TAG(p)) {
    case TAG_NUL: printf("()"); break;
    case TAG_BT0: printf("0");  break;
    case TAG_BT1: printf("1");  break;
    case TAG_TUP: {
      uint16_t loc = LOC(p);
      printf("(");
      print_term(heap_fst(loc));
      printf(",");
      print_term(heap_snd(loc));
      printf(")");
      break;
    }
  }
}

// Main
// ----

int main(int argc, char **argv) {
  memset(heap, 0, sizeof(heap));

  // Pre-seed free list so alloc2 never hits bump-allocator branch
  for (uint32_t i = 2; i < HEAP_SIZE; i += 2) {
    heap[i >> 1] = (uint32_t)free_head;
    free_head    = (uint16_t)i;
  }

  // Program source (UTF-8 λ = \xCE\xBB)
  const char *prog_src =
    "\xCE\xBB! \xCE\xBB{\n"
    "  0: \xCE\xBB! \xCE\xBB{\n"
    "    0: \xCE\xBBx. ~(0,(1,(~(1,(1,x)),(0,()))))\n"
    "    1: \xCE\xBB! \xCE\xBB! \xCE\xBB{\n"
    "      0: \xCE\xBB(). \xCE\xBBzs. zs\n"
    "      1: \xCE\xBB! \xCE\xBB{\n"
    "        0: \xCE\xBBxs. \xCE\xBBzs. ~(0,(1,(xs,(1,(0,zs)))))\n"
    "        1: \xCE\xBBxs. \xCE\xBBzs. ~(0,(0,~(1,(0,(zs,(1,(1,xs)))))))\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  1: \xCE\xBB! \xCE\xBB{\n"
    "    0: \xCE\xBB! \xCE\xBB! \xCE\xBB{\n"
    "      0: \xCE\xBB(). \xCE\xBBys. ys\n"
    "      1: \xCE\xBB! \xCE\xBBx. \xCE\xBBxs. \xCE\xBBys. (1,(x,~(1,(0,(xs,ys)))))\n"
    "    }\n"
    "    1: \xCE\xBB! \xCE\xBB{\n"
    "      0: \xCE\xBB(). (0,())\n"
    "      1: \xCE\xBB! \xCE\xBB{\n"
    "        0: \xCE\xBBxs. (1,(1,~(1,(1,xs))))\n"
    "        1: \xCE\xBBxs. (1,(0,xs))\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";

  // Compile program to bytecode
  {
    Src s = { prog_src, 0, (int)strlen(prog_src) };
    cname_depth = 0;
    code_len    = 0;
    compile_func(&s);
  }

  // Precompute program-level constants
  prog_mat_delta  = code[1] - OP_MAT_BASE;
  // Second-level MAT deltas: branch 0 starts at pc=2 (GET MAT), branch 1 at pc=2+d (GET MAT)
  prog_mat_delta0 = code[3] - OP_MAT_BASE;                       // MAT after GET at pc=2
  prog_mat_delta1 = code[2 + prog_mat_delta + 1] - OP_MAT_BASE;  // MAT after GET at pc=2+d
  memset(fusible, 0, sizeof(fusible));
  for (uint32_t i = 0; i < code_len; i++)
    if (code[i] == OP_REC)
      fusible[i] = is_fusible_arg(code, i + 1);

  // Build FeedSuper table: packed ctrl + delta for each pc
  memset(feed_super, 0, sizeof(feed_super));
  for (uint32_t i = 0; i < code_len; i++) {
    uint8_t op = code[i];
    if (op == OP_GET) {
      uint8_t nxt = (i + 1 < code_len) ? code[i + 1] : 0;
      if (nxt >= OP_MAT_BASE && nxt <= OP_MAT_MAX) {
        feed_super[i] = FEED_PACK(FEED_GET_MAT, nxt - OP_MAT_BASE);
      } else {
        feed_super[i] = FEED_PACK(FEED_GET, 0);
      }
    } else if (op == OP_LAM) {
      feed_super[i] = FEED_PACK(FEED_LAM, 0);
    } else if (op == OP_ERA) {
      feed_super[i] = FEED_PACK(FEED_ERA, 0);
    } else if (op == OP_USE) {
      feed_super[i] = FEED_PACK(FEED_USE, 0);
    } else if (op >= OP_MAT_BASE && op <= OP_MAT_MAX) {
      feed_super[i] = FEED_PACK(FEED_MAT, op - OP_MAT_BASE);
    }
  }

  printf("Bytecode (%u bytes): ", code_len);
  print_code(0, 0);
  printf("\n");

  // Build n = 44 one-bits (LSB-first): (1,(1,...(0,())))
  uint16_t n = PTR_NUL;
  n = make_tup(PTR_BT0, n);
  for (int i = 0; i < 22; i++)
    n = make_tup(PTR_BT1, make_tup(PTR_BT1, n));

  // arg = (0, (0, n)) for ~(0,(0,n)) = main(n)
  uint16_t inner = make_tup(PTR_BT0, n);
  uint16_t arg   = make_tup(PTR_BT0, inner);

  printf("Input built. Starting evaluation...\n");
  fflush(stdout);

  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);

  // Top-level call: feed arg into program, evaluate body
  stat_fun = 1;
  sub_sp   = 0;
  uint32_t body_pc = feed_term(0, arg);
  uint16_t result  = eval(body_pc, 0);

  clock_gettime(CLOCK_MONOTONIC, &ts1);
  double elapsed = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

  // Print result and stats
  print_term(result);
  printf("\n");

  uint64_t total = stat_fun + stat_lam + stat_mat + stat_get + stat_use;
  printf("Interactions: %llu\n", (unsigned long long)total);
  printf("- APP-FUN: %llu\n", (unsigned long long)stat_fun);
  printf("- APP-LAM: %llu\n", (unsigned long long)stat_lam);
  printf("- APP-MAT: %llu\n", (unsigned long long)stat_mat);
  printf("- APP-GET: %llu\n", (unsigned long long)stat_get);
  printf("- APP-USE: %llu\n", (unsigned long long)stat_use);
  printf("Time: %.4f seconds\n", elapsed);
  if (elapsed > 0)
    printf("Speed: %.2f million interactions/second\n", total / elapsed / 1e6);

  return 0;
}
