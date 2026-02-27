//./lambit.ts//

// That works very well - good job!
// Sadly, the TS interpreter is too slow. The term above returns:
//prog: λ! λ{ 0: λ! λ{ 0: λx. ~(0,(1,(~(1,(1,x)),(0,())))); 1: λ! λ! λ{ 0: λ(). λzs. zs; 1: λ! λ{ 0: λxs. λzs. ~(0,(1,(xs,(1,(0,zs))))); 1: λxs. λzs. ~(0,(0,~(1,(0,(zs,(1,(1,xs))))))) } } }; 1: λ! λ{ 0: λ! λ! λ{ 0: λ(). λys. ys; 1: λ! λx. λxs. λys. (1,(x,~(1,(0,(xs,ys))))) }; 1: λ! λ{ 0: λ(). (0,()); 1: λ! λ{ 0: λxs. (1,(1,~(1,(1,xs)))); 1: λxs. (1,(0,xs)) } } } }
//(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(1,(0,(0,())))))))))))))))))))))))))))))))))))))))))
//Interactions: 75496948
//- APP-FUN: 7339984
//- APP-LAM: 11534243
//- APP-MAT: 25165656
//- APP-GET: 30408490
//- APP-USE: 1048575
// In ~8 seconds. That means it achieves only about ~9m interactions/s. 

// Now, your goal is to port LamBit to a fast, memory-efficient C runtime.
// To achieve that, we will represent Terms in memory via U16 Ptrs, where:
// - Ptr ::= Ctr & Loc
// - Ctr ::= NUL | TUP | BT0 | BT1
// - Loc ::= 14-bit address
// We also include a super fast bump allocator, which pre-allocs a buffer with
// 2^14 u16's to store memory terms (using hints to pin it to the L1 cache), and
// works by incrementing the allocation cursor K, and checking if heap[K] is 0.
// We also include a garbage collector, which is triggered whenever a lambda is
// applied to a term, yet the lambda doesn't use it on the returned expression.
// This will zero that term and all its descendants from memory. Note that this
// is only valid if the input program is affine. We assume that's the case.
// Note that, in C, in-memory terms won't include VAR and RET; these belong to
// the Funcs, which are represented via a compact buffer, with 1 byte per node:
// - 0x00      ::= Nul
// - 0x01      ::= Bt0
// - 0x02      ::= Bt1
// - 0x03      ::= Lam(Func)
// - 0x04      ::= Use(Func)
// - 0x05      ::= Get(Func)
// - 0x06      ::= Ret(Func)
// - 0x08~0x1F ::= Var(BIdx)
// - 0x20~0x8F ::= Mat(Func,Func)
// - 0x90~0xFF ::= Tup(Func,Func)
// Here, variables are Bruijn levels (up to 23), and Mat/Tup are represented by
// a range of potential bytes. We use that to embbed the delta position towards
// the second branch/field in the buffer - otherwise, we would not know where to
// locate it.
// The C runtime parses the input source into the compact buffer, and the
// evaluator reads only from it. The normalizer uses a simple substitution
// buffer to store the local context. Our goal is for it to be as fast as
// possible. This will be running in a cluster, so, each flop counts.
// Implement the fast C runtime below:

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define HEAP_SIZE   (1u << 14)
#define CODE_SIZE   (1u << 16)
#define MAX_DEPTH   24
#define MAX_BINDS   256
#define NAME_SIZE   64
#define MAX_SUB     64
#define MAX_FEED    CODE_SIZE
#define MAX_EVAL    CODE_SIZE
#define MAX_FREE    (HEAP_SIZE / 2)

// Set to 1 to enable hot-path runtime checks.
#ifndef LAMBIT_SAFE_CHECKS
#define LAMBIT_SAFE_CHECKS 0
#endif

// Set to 0 to disable interaction counters in hot loops.
#ifndef LAMBIT_COUNT_STATS
#define LAMBIT_COUNT_STATS 1
#endif

// Set to 0 to use the original scanning allocator instead of free-list reuse.
#ifndef LAMBIT_FREE_LIST
#define LAMBIT_FREE_LIST 1
#endif

// Set to 1 to assume affine terms and skip zeroing freed tuple cells.
#ifndef LAMBIT_TRUST_AFFINE
#define LAMBIT_TRUST_AFFINE 1
#endif

// Set to 32 to use 32-bit pointers in the runtime.
#ifndef LAMBIT_PTR_BITS
#define LAMBIT_PTR_BITS 16
#endif

// Set to 1 to force switch dispatch instead of computed-goto.
#ifndef LAMBIT_NO_COMPUTED_GOTO
#define LAMBIT_NO_COMPUTED_GOTO 0
#endif

#if !LAMBIT_NO_COMPUTED_GOTO && (defined(__clang__) || defined(__GNUC__))
#define LAMBIT_COMPUTED_GOTO 1
#else
#define LAMBIT_COMPUTED_GOTO 0
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#define LAMBIT_APPLE_ARM64 1
#else
#define LAMBIT_APPLE_ARM64 0
#endif

#ifndef LAMBIT_COMPUTED_GOTO_FEED
#if LAMBIT_APPLE_ARM64
#define LAMBIT_COMPUTED_GOTO_FEED 0
#else
#define LAMBIT_COMPUTED_GOTO_FEED LAMBIT_COMPUTED_GOTO
#endif
#endif

#ifndef LAMBIT_COMPUTED_GOTO_EVAL
#if LAMBIT_APPLE_ARM64
#define LAMBIT_COMPUTED_GOTO_EVAL 0
#else
#define LAMBIT_COMPUTED_GOTO_EVAL LAMBIT_COMPUTED_GOTO
#endif
#endif

#if LAMBIT_PTR_BITS == 32
typedef uint32_t Ptr;
#else
typedef uint16_t Ptr;
#endif

enum {
  CTR_NUL = 0,
  CTR_TUP = 1,
  CTR_BT0 = 2,
  CTR_BT1 = 3,
};

#define PTR_TAG(p) ((uint16_t)((p) >> 14))
#define PTR_LOC(p) ((uint16_t)((p) & 0x3FFF))
#define MK_PTR(t,l) ((Ptr)((((t) & 0x3) << 14) | ((l) & 0x3FFF)))

static const Ptr PTR_NUL = MK_PTR(CTR_NUL, 1);
static const Ptr PTR_BT0 = MK_PTR(CTR_BT0, 1);
static const Ptr PTR_BT1 = MK_PTR(CTR_BT1, 1);

// Bytecode tags
enum {
  OP_NUL = 0x00,
  OP_BT0 = 0x01,
  OP_BT1 = 0x02,
  OP_LAM = 0x03,
  OP_USE = 0x04,
  OP_GET = 0x05,
  OP_REC = 0x06, // ~term
  OP_ERA = 0x07, // λx. ... where x is unused (compiled from OP_LAM)
  OP_VAR = 0x08, // ..0x1F
  OP_MAT = 0x20, // ..0x8F (delta embedded)
  OP_TUP = 0x90  // ..0xFF (delta embedded)
};

typedef struct {
  uint64_t app_fun, app_lam, app_mat, app_get, app_use;
} Stats;

typedef struct {
  uint8_t  buf[CODE_SIZE];
  bool     lam_used[CODE_SIZE];
  uint32_t len;
} Code;

typedef struct {
  char name[NAME_SIZE];
  int  lam_pc; // bytecode position of binder lambda
} Bind;

typedef struct {
  const char* src;
  size_t      len;
  size_t      idx;
  Code*       out;
  Bind        binds[MAX_BINDS];
  int         bind_len;
} Parser;

typedef struct {
  const Code* code;
  uint32_t    pc;
  uint32_t    rhs_pc;
  uint32_t    base_sp;
  Ptr         lhs;
  uint8_t     mode;
} EvalFrame;

static Ptr   HEAP[HEAP_SIZE] __attribute__((aligned(64)));
static uint16_t HP = 2; // bump cursor
static uint16_t FREE[MAX_FREE];
static uint16_t FREE_SP = 0;
static Stats STATS;
static const Code* PROG_CODE = NULL;
static uint32_t PROG_PC = 0;
static bool PROG_TOP_GET = false;
static Ptr SUB_STACK[MAX_SUB];
static uint32_t SUB_SP = 0;
static Ptr FEED_STACK[MAX_FEED];
static EvalFrame EVAL_STACK[MAX_EVAL];

static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

#if LAMBIT_SAFE_CHECKS
#define HOT_ASSERT(cond, ...) do { if (!(cond)) die(__VA_ARGS__); } while (0)
#else
#define HOT_ASSERT(cond, ...) ((void)0)
#endif

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY(c)   __builtin_expect(!!(c), 1)
#define UNLIKELY(c) __builtin_expect(!!(c), 0)
#else
#define LIKELY(c)   (c)
#define UNLIKELY(c) (c)
#endif

static inline bool is_var(uint8_t op) {
  return (uint8_t)(op - OP_VAR) <= 0x17;
}

static inline bool is_mat(uint8_t op) {
  return (uint8_t)(op - OP_MAT) <= 0x6F;
}

static inline bool is_tup(uint8_t op) {
  return op >= OP_TUP;
}

static inline void free2(uint16_t l) {
#if LAMBIT_FREE_LIST && LAMBIT_TRUST_AFFINE
  HOT_ASSERT(FREE_SP < MAX_FREE, "Tuple free-list overflow.");
  FREE[FREE_SP++] = l;
#else
  HEAP[l + 0] = 0;
  HEAP[l + 1] = 0;
#if LAMBIT_FREE_LIST
  HOT_ASSERT(FREE_SP < MAX_FREE, "Tuple free-list overflow.");
  FREE[FREE_SP++] = l;
#else
  (void)l;
#endif
#endif
}

static void gc_term(Ptr p) {
  if (PTR_TAG(p) != CTR_TUP) return;
  uint16_t l = PTR_LOC(p);
#if !(LAMBIT_FREE_LIST && LAMBIT_TRUST_AFFINE)
  if (HEAP[l + 0] == 0 && HEAP[l + 1] == 0) return;
#endif
  Ptr a = HEAP[l + 0];
  Ptr b = HEAP[l + 1];
  free2(l);
  gc_term(a);
  gc_term(b);
}

static uint16_t alloc2(void) {
#if LAMBIT_FREE_LIST
  if (LIKELY(FREE_SP != 0)) {
    uint16_t l = FREE[--FREE_SP];
#if !LAMBIT_TRUST_AFFINE
    HOT_ASSERT(HEAP[l + 0] == 0 && HEAP[l + 1] == 0, "Corrupt free-list entry.");
#endif
    return l;
  }
  if (LIKELY(HP < HEAP_SIZE - 1)) {
    uint16_t l = HP;
    HP += 2;
    return l;
  }
  die("Out of tuple memory (heap exhausted).");
  return 0;
#else
  uint32_t tries = 0;
  while (tries < HEAP_SIZE) {
    if (HP >= HEAP_SIZE - 1) HP = 2;
    if (HEAP[HP] == 0 && HEAP[HP + 1] == 0) {
      uint16_t l = HP;
      HP += 2;
      return l;
    }
    HP += 2;
    tries += 2;
  }
  die("Out of tuple memory (heap exhausted).");
  return 0;
#endif
}

static inline Ptr mk_tup(Ptr a, Ptr b) {
  uint16_t l = alloc2();
  HEAP[l + 0] = a;
  HEAP[l + 1] = b;
  return MK_PTR(CTR_TUP, l);
}

// ---------------- Parser ----------------

static void skip(Parser* p) {
  for (;;) {
    while (p->idx < p->len && isspace((unsigned char)p->src[p->idx])) p->idx++;
    if (p->idx + 1 < p->len && p->src[p->idx] == '/' && p->src[p->idx + 1] == '/') {
      p->idx += 2;
      while (p->idx < p->len && p->src[p->idx] != '\n') p->idx++;
      continue;
    }
    if (p->idx + 1 < p->len && p->src[p->idx] == '/' && p->src[p->idx + 1] == '*') {
      p->idx += 2;
      while (p->idx + 1 < p->len && !(p->src[p->idx] == '*' && p->src[p->idx + 1] == '/')) p->idx++;
      if (p->idx + 1 < p->len) p->idx += 2;
      continue;
    }
    break;
  }
}

static bool match_str(Parser* p, const char* s) {
  skip(p);
  size_t n = strlen(s);
  if (p->idx + n <= p->len && memcmp(p->src + p->idx, s, n) == 0) {
    p->idx += n;
    return true;
  }
  return false;
}

static void expect_str(Parser* p, const char* s) {
  if (!match_str(p, s)) {
    die("Parse error: expected '%s' at index %zu", s, p->idx);
  }
}

static bool match_lambda(Parser* p) {
  if (match_str(p, "λ")) return true; // UTF-8 lambda
  if (match_str(p, "\\")) return true; // ASCII fallback
  return false;
}

static inline bool is_name_ch(char c) {
  return isalnum((unsigned char)c) || c == '_';
}

static void parse_name(Parser* p, char out[NAME_SIZE]) {
  skip(p);
  size_t i = 0;
  size_t s = p->idx;
  while (p->idx < p->len && is_name_ch(p->src[p->idx])) {
    if (i + 1 < NAME_SIZE) out[i++] = p->src[p->idx];
    p->idx++;
  }
  if (p->idx == s) die("Parse error: expected name at index %zu", p->idx);
  out[i] = 0;
}

static uint32_t emit(Parser* p, uint8_t op) {
  if (p->out->len >= CODE_SIZE) die("Bytecode too large.");
  uint32_t pos = p->out->len++;
  p->out->buf[pos] = op;
  p->out->lam_used[pos] = true;
  return pos;
}

static int find_bind(Parser* p, const char* name, int* out_dist) {
  int dist = 0;
  for (int i = p->bind_len - 1; i >= 0; --i, ++dist) {
    if (strcmp(p->binds[i].name, name) == 0) {
      *out_dist = dist;
      return i;
    }
  }
  return -1;
}

static uint32_t parse_term(Parser* p);
static uint32_t parse_func(Parser* p);

static uint32_t parse_term(Parser* p) {
  skip(p);

  if (match_str(p, "~")) {
    uint32_t pos = emit(p, OP_REC);
    (void)parse_term(p);
    return pos;
  }

  skip(p);
  if (p->idx + 1 < p->len && p->src[p->idx] == '(' && p->src[p->idx + 1] == ')') {
    p->idx += 2;
    return emit(p, OP_NUL);
  }

  skip(p);
  if (p->idx < p->len && p->src[p->idx] == '(' && !(p->idx + 1 < p->len && p->src[p->idx + 1] == ')')) {
    expect_str(p, "(");
    uint32_t pos = emit(p, 0);
    (void)parse_term(p);
    expect_str(p, ",");
    uint32_t right = p->out->len;
    (void)parse_term(p);
    expect_str(p, ")");
    uint32_t d = right - (pos + 1);
    if (d > 0x6F) die("Tuple branch delta too large (%u > 111).", d);
    p->out->buf[pos] = (uint8_t)(OP_TUP + d);
    return pos;
  }

  skip(p);
  if (p->idx < p->len && p->src[p->idx] == '0') {
    char nx = (p->idx + 1 < p->len) ? p->src[p->idx + 1] : 0;
    if (!is_name_ch(nx) && nx != '{') {
      p->idx++;
      return emit(p, OP_BT0);
    }
  }

  skip(p);
  if (p->idx < p->len && p->src[p->idx] == '1') {
    char nx = (p->idx + 1 < p->len) ? p->src[p->idx + 1] : 0;
    if (!is_name_ch(nx) && nx != '{') {
      p->idx++;
      return emit(p, OP_BT1);
    }
  }

  char nam[NAME_SIZE];
  parse_name(p, nam);
  int dist = 0;
  int idx = find_bind(p, nam, &dist);
  if (idx < 0) die("Unbound variable '%s'.", nam);
  if (dist > 23) die("Bruijn index too large for 5-bit encoding (%d).", dist);

  int lam_pc = p->binds[idx].lam_pc;
  if (lam_pc >= 0) p->out->lam_used[lam_pc] = true;
  return emit(p, (uint8_t)(OP_VAR + dist));
}

static uint32_t parse_func(Parser* p) {
  skip(p);

  if (match_lambda(p)) {
    if (match_str(p, "!")) {
      uint32_t pos = emit(p, OP_GET);
      (void)parse_func(p);
      return pos;
    }

    if (match_str(p, "{")) {
      expect_str(p, "0");
      expect_str(p, ":");
      uint32_t pos = emit(p, 0);
      (void)parse_func(p);
      (void)match_str(p, ";");
      expect_str(p, "1");
      expect_str(p, ":");
      uint32_t right = p->out->len;
      (void)parse_func(p);
      (void)match_str(p, ";");
      expect_str(p, "}");
      uint32_t d = right - (pos + 1);
      if (d > 0x6F) die("Match branch delta too large (%u > 111).", d);
      p->out->buf[pos] = (uint8_t)(OP_MAT + d);
      return pos;
    }

    if (match_str(p, "()")) {
      expect_str(p, ".");
      uint32_t pos = emit(p, OP_USE);
      (void)parse_func(p);
      return pos;
    }

    char nam[NAME_SIZE];
    parse_name(p, nam);
    expect_str(p, ".");

    uint32_t pos = emit(p, OP_LAM);
    p->out->lam_used[pos] = false;

    if (p->bind_len >= MAX_BINDS) die("Too many nested binders.");
    strncpy(p->binds[p->bind_len].name, nam, NAME_SIZE - 1);
    p->binds[p->bind_len].name[NAME_SIZE - 1] = 0;
    p->binds[p->bind_len].lam_pc = (int)pos;
    p->bind_len++;

    (void)parse_func(p);

    p->bind_len--;
    return pos;
  }

  // Ret ::= Term (implicit in bytecode: term node itself)
  return parse_term(p);
}

static uint32_t compile_func(Code* out, const char* src) {
  memset(out, 0, sizeof(*out));
  Parser p = {0};
  p.src = src;
  p.len = strlen(src);
  p.idx = 0;
  p.out = out;
  p.bind_len = 0;
  uint32_t root = parse_func(&p);
  skip(&p);
  if (p.idx != p.len) die("Trailing input after func at index %zu.", p.idx);

  // Specialize unused lambdas into ERA nodes to remove a hot runtime branch.
  for (uint32_t i = 0; i < out->len; ++i) {
    if (out->buf[i] == OP_LAM && !out->lam_used[i]) {
      out->buf[i] = OP_ERA;
    }
  }

  return root;
}

static uint32_t compile_term(Code* out, const char* src) {
  memset(out, 0, sizeof(*out));
  Parser p = {0};
  p.src = src;
  p.len = strlen(src);
  p.idx = 0;
  p.out = out;
  p.bind_len = 0;
  uint32_t root = parse_term(&p);
  skip(&p);
  if (p.idx != p.len) die("Trailing input after term at index %zu.", p.idx);
  return root;
}

// ---------------- Eval ----------------

static Ptr eval_term(const Code* c, uint32_t pc);

// Feeds one term argument into a function starting at `pc`.
static inline __attribute__((always_inline))
uint32_t feed_term(
  uint32_t pc,
  Ptr arg,
  uint32_t* restrict sub_sp_p,
  uint64_t* restrict app_lam_p,
  uint64_t* restrict app_mat_p,
  uint64_t* restrict app_get_p,
  uint64_t* restrict app_use_p
) {
  uint8_t op = 0;
  uint32_t feed_sp = 0;
  uint32_t sub_sp = *sub_sp_p;
  uint64_t app_lam = *app_lam_p;
  uint64_t app_mat = *app_mat_p;
  uint64_t app_get = *app_get_p;
  uint64_t app_use = *app_use_p;

#if LAMBIT_COMPUTED_GOTO_FEED
  static bool init = false;
  static void* jump_tbl[256];
  if (UNLIKELY(!init)) {
    for (uint32_t i = 0; i < 256; ++i) jump_tbl[i] = &&L_BAD;
    jump_tbl[OP_GET] = &&L_GET;
    jump_tbl[OP_LAM] = &&L_LAM;
    jump_tbl[OP_ERA] = &&L_ERA;
    jump_tbl[OP_USE] = &&L_USE;
    for (uint32_t i = OP_MAT; i <= 0x8F; ++i) jump_tbl[i] = &&L_MAT;
    init = true;
  }

#define FEED_DISPATCH() do { \
    op = PROG_CODE->buf[pc]; \
    goto *jump_tbl[op]; \
  } while (0)
#else
#define FEED_DISPATCH() do { \
    op = PROG_CODE->buf[pc]; \
    goto L_SWITCH; \
  } while (0)
#endif

#define FEED_POP_OR_RET() do { \
    if (feed_sp == 0) { \
      *sub_sp_p = sub_sp; \
      *app_lam_p = app_lam; \
      *app_mat_p = app_mat; \
      *app_get_p = app_get; \
      *app_use_p = app_use; \
      return pc; \
    } \
    arg = FEED_STACK[--feed_sp]; \
    FEED_DISPATCH(); \
  } while (0)

  FEED_DISPATCH();

#if !LAMBIT_COMPUTED_GOTO_FEED
L_SWITCH: {
    if (op == OP_GET) goto L_GET;
    if (is_mat(op)) goto L_MAT;
    if (op == OP_LAM) goto L_LAM;
    if (op == OP_ERA) goto L_ERA;
    if (op == OP_USE) goto L_USE;
    goto L_BAD;
  }
#endif

L_GET: {
    HOT_ASSERT(PTR_TAG(arg) == CTR_TUP, "Get expected tuple.");
    HOT_ASSERT(feed_sp < MAX_FEED, "Feed stack overflow.");
#if LAMBIT_COUNT_STATS
    app_get++;
#endif
    uint16_t l = PTR_LOC(arg);
    Ptr a = HEAP[l + 0];
    Ptr b = HEAP[l + 1];
    free2(l);
    FEED_STACK[feed_sp++] = b;
    pc = pc + 1;
    arg = a;
    FEED_DISPATCH();
  }

L_LAM: {
#if LAMBIT_COUNT_STATS
    app_lam++;
#endif
    HOT_ASSERT(sub_sp < MAX_SUB, "Substitution stack overflow.");
    SUB_STACK[sub_sp++] = arg;
    pc = pc + 1;
    FEED_POP_OR_RET();
  }

L_ERA: {
#if LAMBIT_COUNT_STATS
    app_lam++;
#endif
    HOT_ASSERT(sub_sp < MAX_SUB, "Substitution stack overflow.");
    gc_term(arg);
    SUB_STACK[sub_sp++] = PTR_NUL;
    pc = pc + 1;
    FEED_POP_OR_RET();
  }

L_USE: {
    HOT_ASSERT(PTR_TAG(arg) == CTR_NUL, "Use expected ().");
#if LAMBIT_COUNT_STATS
    app_use++;
#endif
    pc = pc + 1;
    FEED_POP_OR_RET();
  }

L_MAT: {
    HOT_ASSERT(PTR_TAG(arg) == CTR_BT0 || PTR_TAG(arg) == CTR_BT1, "Mat expected 0/1 bit.");
#if LAMBIT_COUNT_STATS
    app_mat++;
#endif
    uint32_t d = (uint32_t)(op - OP_MAT);
    pc = (arg == PTR_BT0) ? (pc + 1) : (pc + 1 + d);
    FEED_POP_OR_RET();
  }

L_BAD: {
    die("Cannot apply non-function opcode 0x%02X at pc=%u.", op, pc);
  }

#undef FEED_POP_OR_RET
#undef FEED_DISPATCH
  return 0;
}

enum {
  EVAL_ENTER   = 0,
  EVAL_TUP_LHS = 1,
  EVAL_TUP_RHS = 2,
  EVAL_REC_ARG = 3,
};

// Evaluates a term to normal form.
static Ptr eval_term(const Code* c, uint32_t pc) {
  uint8_t op = 0;
  Ptr ret = PTR_NUL;
  uint32_t eval_sp = 0;
  EvalFrame* frm = NULL;
  uint32_t sub_sp = SUB_SP;
  uint64_t app_fun = STATS.app_fun;
  uint64_t app_lam = STATS.app_lam;
  uint64_t app_mat = STATS.app_mat;
  uint64_t app_get = STATS.app_get;
  uint64_t app_use = STATS.app_use;

  EVAL_STACK[eval_sp++] = (EvalFrame){
    .code = c,
    .pc = pc,
    .base_sp = sub_sp,
  };

#if LAMBIT_COMPUTED_GOTO_EVAL
  static bool init = false;
  static void* jump_tbl[256];
  if (UNLIKELY(!init)) {
    for (uint32_t i = 0; i < 256; ++i) jump_tbl[i] = &&L_BAD;
    jump_tbl[OP_NUL] = &&L_NUL;
    jump_tbl[OP_BT0] = &&L_BT0;
    jump_tbl[OP_BT1] = &&L_BT1;
    jump_tbl[OP_REC] = &&L_REC;
    for (uint32_t i = OP_VAR; i <= 0x1F; ++i) jump_tbl[i] = &&L_VAR;
    for (uint32_t i = OP_TUP; i <= 0xFF; ++i) jump_tbl[i] = &&L_TUP;
    init = true;
  }

#define EVAL_DISPATCH() do { \
    frm = &EVAL_STACK[eval_sp - 1]; \
    op = frm->code->buf[frm->pc]; \
    goto *jump_tbl[op]; \
  } while (0)
#else
#define EVAL_DISPATCH() do { \
    frm = &EVAL_STACK[eval_sp - 1]; \
    op = frm->code->buf[frm->pc]; \
    goto L_SWITCH; \
  } while (0)
#endif

#define EVAL_PUSH(next_code, next_pc) do { \
    HOT_ASSERT(eval_sp < MAX_EVAL, "Eval stack overflow."); \
    EVAL_STACK[eval_sp].code = (next_code); \
    EVAL_STACK[eval_sp].pc = (next_pc); \
    EVAL_STACK[eval_sp].base_sp = sub_sp; \
    eval_sp++; \
  } while (0)

#define EVAL_RET(val) do { \
    ret = (val); \
    frm = &EVAL_STACK[eval_sp - 1]; \
    sub_sp = frm->base_sp; \
    eval_sp--; \
    if (eval_sp == 0) { \
      SUB_SP = sub_sp; \
      STATS.app_fun = app_fun; \
      STATS.app_lam = app_lam; \
      STATS.app_mat = app_mat; \
      STATS.app_get = app_get; \
      STATS.app_use = app_use; \
      return ret; \
    } \
    goto L_CONT; \
  } while (0)

  EVAL_DISPATCH();

#if !LAMBIT_COMPUTED_GOTO_EVAL
L_SWITCH: {
    if (is_tup(op)) goto L_TUP;
    if (is_var(op)) goto L_VAR;
    if (op == OP_REC) goto L_REC;
    if (op == OP_NUL) goto L_NUL;
    if (op == OP_BT0) goto L_BT0;
    if (op == OP_BT1) goto L_BT1;
    goto L_BAD;
  }
#endif

L_NUL: {
    EVAL_RET(PTR_NUL);
  }

L_BT0: {
    EVAL_RET(PTR_BT0);
  }

L_BT1: {
    EVAL_RET(PTR_BT1);
  }

L_VAR: {
    uint8_t idx = (uint8_t)(op - OP_VAR);
    HOT_ASSERT(idx < sub_sp, "Variable out of scope: idx=%u sub_sp=%u", idx, sub_sp);
    Ptr val = SUB_STACK[sub_sp - 1 - idx];
    EVAL_RET(val);
  }

L_TUP: {
    uint32_t d = (uint32_t)(op - OP_TUP);
    frm->rhs_pc = frm->pc + 1 + d;
    frm->mode = EVAL_TUP_LHS;
    EVAL_PUSH(frm->code, frm->pc + 1);
    EVAL_DISPATCH();
  }

L_REC: {
#if LAMBIT_COUNT_STATS
    app_fun++;
#endif
    frm->mode = EVAL_REC_ARG;
    EVAL_PUSH(frm->code, frm->pc + 1);
    EVAL_DISPATCH();
  }

L_CONT: {
    frm = &EVAL_STACK[eval_sp - 1];
    switch (frm->mode) {
      case EVAL_TUP_LHS: {
        frm->lhs = ret;
        frm->mode = EVAL_TUP_RHS;
        EVAL_PUSH(frm->code, frm->rhs_pc);
        EVAL_DISPATCH();
      }
      case EVAL_TUP_RHS: {
        if (PROG_TOP_GET && eval_sp >= 2) {
          EvalFrame* par = &EVAL_STACK[eval_sp - 2];
          if (par->mode == EVAL_REC_ARG) {
            sub_sp = 0;
#if LAMBIT_COUNT_STATS
            app_get++;
#endif
            uint32_t body_pc = feed_term(
              PROG_PC + 1, frm->lhs, &sub_sp, &app_lam, &app_mat, &app_get, &app_use
            );
            body_pc = feed_term(
              body_pc, ret, &sub_sp, &app_lam, &app_mat, &app_get, &app_use
            );
            par->code = PROG_CODE;
            par->pc = body_pc;
            par->mode = EVAL_ENTER;
            eval_sp--;
            EVAL_DISPATCH();
          }
        }
        Ptr tup = mk_tup(frm->lhs, ret);
        EVAL_RET(tup);
      }
      case EVAL_REC_ARG: {
        sub_sp = 0;
        frm->code = PROG_CODE;
        frm->pc = feed_term(
          PROG_PC, ret, &sub_sp, &app_lam, &app_mat, &app_get, &app_use
        );
        EVAL_DISPATCH();
      }
      default: {
        die("Invalid eval continuation mode %u.", (unsigned)frm->mode);
      }
    }
  }

L_BAD: {
    uint32_t bad_pc = frm ? frm->pc : pc;
    die("Expected term opcode, got 0x%02X at pc=%u.", op, bad_pc);
  }

#undef EVAL_RET
#undef EVAL_PUSH
#undef EVAL_DISPATCH
  return PTR_NUL;
}

// ---------------- Show ----------------

static void show_ptr(Ptr p) {
  switch (PTR_TAG(p)) {
    case CTR_NUL: printf("()"); return;
    case CTR_BT0: printf("0"); return;
    case CTR_BT1: printf("1"); return;
    case CTR_TUP: {
      uint16_t l = PTR_LOC(p);
      printf("(");
      show_ptr(HEAP[l + 0]);
      printf(",");
      show_ptr(HEAP[l + 1]);
      printf(")");
      return;
    }
    default: die("Invalid pointer tag.");
  }
}

static void show_stats(const Stats* s) {
  uint64_t total = s->app_fun + s->app_lam + s->app_mat + s->app_get + s->app_use;
  printf("Interactions: %llu\n", (unsigned long long)total);
  printf("- APP-FUN: %llu\n", (unsigned long long)s->app_fun);
  printf("- APP-LAM: %llu\n", (unsigned long long)s->app_lam);
  printf("- APP-MAT: %llu\n", (unsigned long long)s->app_mat);
  printf("- APP-GET: %llu\n", (unsigned long long)s->app_get);
  printf("- APP-USE: %llu\n", (unsigned long long)s->app_use);
}

// ---------------- Main ----------------

int main(int argc, char** argv) {
  memset(HEAP, 0, sizeof(HEAP));
  memset(&STATS, 0, sizeof(STATS));
  memset(SUB_STACK, 0, sizeof(SUB_STACK));

  const char* prog_src = NULL;
  char* input_heap = NULL;

  if (argc == 3) {
    prog_src = argv[1];
    input_heap = strdup(argv[2]);
  } else {
    prog_src =
      "λ! λ{\n"
      "  0: λ! λ{\n"
      "    0: λx. ~(0,(1,(~(1,(1,x)),(0,()))))\n"
      "    1: λ! λ! λ{\n"
      "      0: λ(). λzs. zs\n"
      "      1: λ! λ{\n"
      "        0: λxs. λzs. ~(0,(1,(xs,(1,(0,zs)))))\n"
      "        1: λxs. λzs. ~(0,(0,~(1,(0,(zs,(1,(1,xs)))))))\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "  1: λ! λ{\n"
      "    0: λ! λ! λ{\n"
      "      0: λ(). λys. ys\n"
      "      1: λ! λx. λxs. λys. (1,(x,~(1,(0,(xs,ys)))))\n"
      "    }\n"
      "    1: λ! λ{\n"
      "      0: λ(). (0,())\n"
      "      1: λ! λ{\n"
      "        0: λxs. (1,(1,~(1,(1,xs))))\n"
      "        1: λxs. (1,(0,xs))\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "}";

    char* n = strdup("(0,())");
    for (int i = 0; i < 22; ++i) {
      size_t sz = strlen(n) + 16;
      char* nn = (char*)malloc(sz);
      snprintf(nn, sz, "(1,(1,%s))", n);
      free(n);
      n = nn;
    }
    size_t isz = strlen(n) + 16;
    input_heap = (char*)malloc(isz);
    snprintf(input_heap, isz, "~(0,(0,%s))", n);
    free(n);
  }

  Code prog = {0};
  Code input = {0};

  PROG_PC = compile_func(&prog, prog_src);
  PROG_CODE = &prog;
  PROG_TOP_GET = (prog.buf[PROG_PC] == OP_GET);
  uint32_t input_pc = compile_term(&input, input_heap);

  SUB_SP = 0;
  Ptr result = eval_term(&input, input_pc);
  show_ptr(result);
  printf("\n");
  show_stats(&STATS);

  free(input_heap);
  return 0;
}
