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
#define HEAP_PAIRS  (HEAP_SIZE / 2)
#define CODE_SIZE   (1u << 16)
#define MAX_DEPTH   24
#define MAX_BINDS   256
#define NAME_SIZE   64
#define MAX_SUB     64
#define MAX_FEED    CODE_SIZE
#define MAX_EVAL    CODE_SIZE

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

#if LAMBIT_PTR_BITS == 32
typedef uint64_t Cont;
#else
typedef uint32_t Cont;
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

#if LAMBIT_PTR_BITS == 16
static uint32_t HEAP[HEAP_PAIRS] __attribute__((aligned(64)));
#else
static Ptr HEAP[HEAP_SIZE] __attribute__((aligned(64)));
#endif
#if LAMBIT_PTR_BITS == 16
static uint16_t HP = 1; // bump cursor in pair indices
#else
static uint16_t HP = 2; // bump cursor in cell indices
#endif
static uint16_t FREE_HEAD = 0;
static Stats STATS;
static const Code* PROG_CODE = NULL;
static uint32_t PROG_PC = 0;
static bool PROG_TOP_GET = false;
static bool PROG_TOP_GET_MAT = false;
static uint32_t PROG_TOP_MAT_DELTA = 0;
static Ptr SUB_STACK[MAX_SUB] __attribute__((aligned(64)));
static uint32_t SUB_SP = 0;
static Ptr FEED_STACK[MAX_FEED] __attribute__((aligned(64)));
static Cont EVAL_CONT_STACK[MAX_EVAL] __attribute__((aligned(64)));

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

static inline Ptr heap_get_fst(uint16_t l) {
#if LAMBIT_PTR_BITS == 16
  return (Ptr)(HEAP[l] & 0xFFFFu);
#else
  return HEAP[l + 0];
#endif
}

static inline Ptr heap_get_snd(uint16_t l) {
#if LAMBIT_PTR_BITS == 16
  return (Ptr)(HEAP[l] >> 16);
#else
  return HEAP[l + 1];
#endif
}

static inline void heap_get_pair(uint16_t l, Ptr* fst, Ptr* snd) {
#if LAMBIT_PTR_BITS == 16
  uint32_t pair = HEAP[l];
  *fst = (Ptr)(pair & 0xFFFFu);
  *snd = (Ptr)(pair >> 16);
#else
  *fst = HEAP[l + 0];
  *snd = HEAP[l + 1];
#endif
}

static inline void heap_set_pair(uint16_t l, Ptr a, Ptr b) {
#if LAMBIT_PTR_BITS == 16
  HEAP[l] = ((uint32_t)b << 16) | (uint32_t)a;
#else
  HEAP[l + 0] = a;
  HEAP[l + 1] = b;
#endif
}

static inline __attribute__((unused))
bool heap_pair_is_zero(uint16_t l) {
#if LAMBIT_PTR_BITS == 16
  return HEAP[l] == 0;
#else
  return HEAP[l + 0] == 0 && HEAP[l + 1] == 0;
#endif
}

static inline uint16_t heap_get_free_next(uint16_t l) {
#if LAMBIT_PTR_BITS == 16
  return (uint16_t)(HEAP[l] & 0xFFFFu);
#else
  return (uint16_t)HEAP[l + 0];
#endif
}

static inline void heap_set_free_next(uint16_t l, uint16_t nxt) {
#if LAMBIT_PTR_BITS == 16
  HEAP[l] = (uint32_t)nxt;
#else
  HEAP[l + 0] = nxt;
#endif
}

static inline void free2(uint16_t l) {
#if LAMBIT_FREE_LIST && LAMBIT_TRUST_AFFINE
  heap_set_free_next(l, FREE_HEAD);
  FREE_HEAD = l;
#else
  heap_set_pair(l, 0, 0);
#if LAMBIT_FREE_LIST
  heap_set_free_next(l, FREE_HEAD);
  FREE_HEAD = l;
#else
  (void)l;
#endif
#endif
}

static void gc_term(Ptr p) {
  if (PTR_TAG(p) != CTR_TUP) return;
  uint16_t l = PTR_LOC(p);
#if !(LAMBIT_FREE_LIST && LAMBIT_TRUST_AFFINE)
  if (heap_pair_is_zero(l)) return;
#endif
  Ptr a;
  Ptr b;
  heap_get_pair(l, &a, &b);
  free2(l);
  gc_term(a);
  gc_term(b);
}

static uint16_t alloc2(void) {
#if LAMBIT_FREE_LIST
  if (LIKELY(FREE_HEAD != 0)) {
    uint16_t l = FREE_HEAD;
    FREE_HEAD = heap_get_free_next(l);
#if !LAMBIT_TRUST_AFFINE
    HOT_ASSERT(heap_pair_is_zero(l), "Corrupt free-list entry.");
#endif
    return l;
  }
  #if LAMBIT_PTR_BITS == 16
  if (LIKELY(HP < HEAP_PAIRS)) {
    uint16_t l = HP;
    HP += 1;
    return l;
  }
  #else
  if (LIKELY(HP < HEAP_SIZE - 1)) {
    uint16_t l = HP;
    HP += 2;
    return l;
  }
  #endif
  die("Out of tuple memory (heap exhausted).");
  return 0;
#else
  uint32_t tries = 0;
  #if LAMBIT_PTR_BITS == 16
  while (tries < HEAP_PAIRS) {
    if (HP >= HEAP_PAIRS) HP = 1;
    if (heap_pair_is_zero(HP)) {
      uint16_t l = HP;
      HP += 1;
      return l;
    }
    HP += 1;
    tries += 1;
  }
  #else
  while (tries < HEAP_SIZE) {
    if (HP >= HEAP_SIZE - 1) HP = 2;
    if (heap_pair_is_zero(HP)) {
      uint16_t l = HP;
      HP += 2;
      return l;
    }
    HP += 2;
    tries += 2;
  }
  #endif
  die("Out of tuple memory (heap exhausted).");
  return 0;
#endif
}

static inline Ptr mk_tup(Ptr a, Ptr b) {
  uint16_t l = alloc2();
  heap_set_pair(l, a, b);
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
static inline __attribute__((always_inline, hot))
uint32_t feed_term(
  uint32_t pc,
  Ptr arg,
  uint32_t* restrict sub_sp_p,
  uint64_t* restrict app_lam_p,
  uint64_t* restrict app_mat_p,
  uint64_t* restrict app_get_p,
  uint64_t* restrict app_use_p
) {
  Ptr* const feed_base = FEED_STACK;
  Ptr* feed_top = feed_base;
  uint32_t sub_sp = *sub_sp_p;
  Ptr* restrict sub_stack = SUB_STACK;
  const uint8_t* cc = PROG_CODE->buf;
  uint64_t app_lam = *app_lam_p;
  uint64_t app_mat = *app_mat_p;
  uint64_t app_get = *app_get_p;
  uint64_t app_use = *app_use_p;

  for (;;) {
    uint8_t op = cc[pc];

    if (LIKELY(op == OP_GET)) {
      HOT_ASSERT(PTR_TAG(arg) == CTR_TUP, "Get expected tuple.");
      HOT_ASSERT(feed_top < feed_base + MAX_FEED, "Feed stack overflow.");
#if LAMBIT_COUNT_STATS
      app_get++;
#endif
      uint16_t l = PTR_LOC(arg);
      Ptr a;
      Ptr b;
      heap_get_pair(l, &a, &b);
      free2(l);

      uint8_t nxt = cc[pc + 1];
      if (is_mat(nxt)) {
#if LAMBIT_COUNT_STATS
        app_mat++;
#endif
        uint32_t d = (uint32_t)(nxt - OP_MAT);
        pc = (a == PTR_BT0) ? (pc + 2) : (pc + 2 + d);
        arg = b;
        continue;
      }

      *feed_top++ = b;
      pc = pc + 1;
      arg = a;
      continue;
    }

    if (LIKELY(op == OP_LAM)) {
#if LAMBIT_COUNT_STATS
      app_lam++;
#endif
      HOT_ASSERT(sub_sp < MAX_SUB, "Substitution stack overflow.");
      sub_stack[sub_sp++] = arg;
      pc = pc + 1;
      if (feed_top == feed_base) break;
      arg = *--feed_top;
      continue;
    }

    if (LIKELY(op == OP_USE)) {
      HOT_ASSERT(PTR_TAG(arg) == CTR_NUL, "Use expected ().");
#if LAMBIT_COUNT_STATS
      app_use++;
#endif
      pc = pc + 1;
      if (feed_top == feed_base) break;
      arg = *--feed_top;
      continue;
    }

    if (UNLIKELY(op == OP_ERA)) {
#if LAMBIT_COUNT_STATS
      app_lam++;
#endif
      HOT_ASSERT(sub_sp < MAX_SUB, "Substitution stack overflow.");
      if (PTR_TAG(arg) == CTR_TUP) {
        gc_term(arg);
      }
      sub_stack[sub_sp++] = PTR_NUL;
      pc = pc + 1;
      if (feed_top == feed_base) break;
      arg = *--feed_top;
      continue;
    }

    if (UNLIKELY(is_mat(op))) {
      HOT_ASSERT(PTR_TAG(arg) == CTR_BT0 || PTR_TAG(arg) == CTR_BT1, "Mat expected 0/1 bit.");
#if LAMBIT_COUNT_STATS
      app_mat++;
#endif
      uint32_t d = (uint32_t)(op - OP_MAT);
      pc = (arg == PTR_BT0) ? (pc + 1) : (pc + 1 + d);
      if (feed_top == feed_base) break;
      arg = *--feed_top;
      continue;
    }

    die("Cannot apply non-function opcode 0x%02X at pc=%u.", op, pc);
  }

  *sub_sp_p = sub_sp;
  *app_lam_p = app_lam;
  *app_mat_p = app_mat;
  *app_get_p = app_get;
  *app_use_p = app_use;
  return pc;
}

enum {
  CONT_TUP_SND  = 1,
  CONT_TUP_DONE = 2,
  CONT_REC_FEED = 3,
  CONT_REC_DONE = 4,
};

#if LAMBIT_PTR_BITS == 16
#define CONT_MAKE(tag_v, code_id_v, saved_sp_v, payload_v) \
  ((((uint32_t)(tag_v)) << 25) | \
   (((uint32_t)(code_id_v)) << 24) | \
   (((uint32_t)(saved_sp_v)) << 16) | \
   ((uint32_t)(payload_v)))

#define CONT_TAG(c)      ((uint8_t)((c) >> 25))
#define CONT_CODE_ID(c)  ((uint8_t)(((c) >> 24) & 0x01u))
#define CONT_SAVED_SP(c) ((uint8_t)((c) >> 16))
#define CONT_PAYLOAD(c)  ((uint32_t)(uint16_t)(c))
#else
#define CONT_MAKE(tag_v, code_id_v, saved_sp_v, payload_v) \
  ((((uint64_t)(tag_v)) << 48) | \
   (((uint64_t)(code_id_v)) << 40) | \
   (((uint64_t)(saved_sp_v)) << 32) | \
   ((uint64_t)(payload_v)))

#define CONT_TAG(c)      ((uint8_t)((c) >> 48))
#define CONT_CODE_ID(c)  ((uint8_t)((c) >> 40))
#define CONT_SAVED_SP(c) ((uint8_t)((c) >> 32))
#define CONT_PAYLOAD(c)  ((uint32_t)(c))
#endif

// Evaluates a term to normal form.
static __attribute__((hot, flatten))
Ptr eval_term(const Code* c, uint32_t start_pc) {
  enum {
    CODE_INPUT = 0,
    CODE_PROG  = 1,
  };

  uint8_t op = 0;
  Ptr val = PTR_NUL;
  uint32_t pc = start_pc;
  uint8_t code_id = CODE_INPUT;
  const uint8_t* cc = c->buf;
  uint32_t cont_sp = 0;
  Cont* restrict cont_stack = EVAL_CONT_STACK;
  Ptr* restrict sub_stack = SUB_STACK;
  uint32_t sub_sp = SUB_SP;
  uint64_t app_fun = STATS.app_fun;
  uint64_t app_lam = STATS.app_lam;
  uint64_t app_mat = STATS.app_mat;
  uint64_t app_get = STATS.app_get;
  uint64_t app_use = STATS.app_use;
  const bool top_get = PROG_TOP_GET;
  const bool top_get_mat = PROG_TOP_GET_MAT;
  const uint32_t top_pc = PROG_PC;
  const uint32_t top_mat_delta = PROG_TOP_MAT_DELTA;
#define CONT_PUSH(tag, cid, ssp, pay) do { \
    HOT_ASSERT(cont_sp < MAX_EVAL, "Eval continuation stack overflow."); \
    cont_stack[cont_sp++] = CONT_MAKE((tag), (cid), (ssp), (pay)); \
  } while (0)

L_EVAL: {
    op = cc[pc];

    if (LIKELY(is_tup(op))) goto L_TUP;
    if (is_var(op)) goto L_VAR;

    if (op == OP_REC) {
#if LAMBIT_COUNT_STATS
      app_fun++;
#endif
      if (cont_sp > 0) {
        Cont c0 = cont_stack[cont_sp - 1];
        if (CONT_TAG(c0) == CONT_REC_DONE) {
          uint32_t saved_sp = CONT_SAVED_SP(c0);
          cont_stack[cont_sp - 1] = CONT_MAKE(CONT_REC_FEED, 0, saved_sp, 0);
          pc = pc + 1;
          goto L_EVAL;
        }
      }
      CONT_PUSH(CONT_REC_FEED, 0, sub_sp, 0);
      pc = pc + 1;
      goto L_EVAL;
    }
    if (op == OP_NUL) {
      val = PTR_NUL;
      goto L_RET;
    }
    if (op == OP_BT0) {
      val = PTR_BT0;
      goto L_RET;
    }
    if (op == OP_BT1) {
      val = PTR_BT1;
      goto L_RET;
    }
    goto L_BAD;
  }

L_VAR: {
    uint8_t idx = (uint8_t)(op - OP_VAR);
    HOT_ASSERT(idx < sub_sp, "Variable out of scope: idx=%u sub_sp=%u", idx, sub_sp);
    val = sub_stack[sub_sp - 1 - idx];
    goto L_RET;
  }

L_TUP: {
    uint32_t d = (uint32_t)(op - OP_TUP);
    if (LIKELY(d == 1)) {
      uint8_t fst_op = cc[pc + 1];
      Ptr fst_val = PTR_NUL;

      if (LIKELY(fst_op == OP_BT1)) {
        fst_val = PTR_BT1;
      } else if (fst_op == OP_BT0) {
        fst_val = PTR_BT0;
      } else if (is_var(fst_op)) {
        uint8_t idx = (uint8_t)(fst_op - OP_VAR);
        HOT_ASSERT(idx < sub_sp, "Variable out of scope: idx=%u sub_sp=%u", idx, sub_sp);
        fst_val = sub_stack[sub_sp - 1 - idx];
      } else if (fst_op == OP_NUL) {
        fst_val = PTR_NUL;
      } else {
        goto L_TUP_SLOW;
      }

      CONT_PUSH(CONT_TUP_DONE, 0, sub_sp, fst_val);
      pc = pc + 2;
      goto L_EVAL;
    }

L_TUP_SLOW:
    CONT_PUSH(CONT_TUP_SND, code_id, sub_sp, pc + 1 + d);
    pc = pc + 1;
    goto L_EVAL;
  }

L_RET: {
    if (cont_sp == 0) goto L_DONE;
    Cont c0 = cont_stack[cont_sp - 1];
    uint8_t tag = CONT_TAG(c0);

    if (LIKELY(tag == CONT_TUP_DONE)) {
      if (LIKELY(top_get_mat)) {
        for (;;) {
          if (LIKELY(cont_sp >= 2)) {
            Cont c1 = cont_stack[cont_sp - 2];
            uint8_t t1 = CONT_TAG(c1);
            if (LIKELY(t1 == CONT_TUP_DONE)) {
              val = mk_tup((Ptr)CONT_PAYLOAD(c0), val);
              cont_sp = cont_sp - 1;
              c0 = c1;
              continue;
            }
            if (t1 == CONT_REC_FEED) {
              uint32_t saved_sp = CONT_SAVED_SP(c1);
              sub_sp = 0;
#if LAMBIT_COUNT_STATS
              app_get++;
              app_mat++;
#endif
              Ptr fst = (Ptr)CONT_PAYLOAD(c0);
              uint32_t body_pc = (fst == PTR_BT0)
                ? (top_pc + 2)
                : (top_pc + 2 + top_mat_delta);
              body_pc = feed_term(
                body_pc, val, &sub_sp, &app_lam, &app_mat, &app_get, &app_use
              );
              cont_stack[cont_sp - 2] = CONT_MAKE(CONT_REC_DONE, 0, saved_sp, 0);
              cont_sp = cont_sp - 1;
              code_id = CODE_PROG;
              cc = PROG_CODE->buf;
              pc = body_pc;
              goto L_EVAL;
            }
          }

          sub_sp = CONT_SAVED_SP(c0);
          val = mk_tup((Ptr)CONT_PAYLOAD(c0), val);
          cont_sp = cont_sp - 1;
          if (cont_sp == 0) goto L_DONE;
          goto L_RET;
        }
      }

      for (;;) {
        if (top_get && cont_sp >= 2) {
          Cont c1 = cont_stack[cont_sp - 2];
          if (CONT_TAG(c1) == CONT_REC_FEED) {
            uint32_t saved_sp = CONT_SAVED_SP(c1);
            sub_sp = 0;
#if LAMBIT_COUNT_STATS
            app_get++;
#endif
            uint32_t body_pc = feed_term(
              top_pc + 1, (Ptr)CONT_PAYLOAD(c0), &sub_sp, &app_lam, &app_mat, &app_get, &app_use
            );
            body_pc = feed_term(
              body_pc, val, &sub_sp, &app_lam, &app_mat, &app_get, &app_use
            );
            cont_stack[cont_sp - 2] = CONT_MAKE(CONT_REC_DONE, 0, saved_sp, 0);
            cont_sp = cont_sp - 1;
            code_id = CODE_PROG;
            cc = PROG_CODE->buf;
            pc = body_pc;
            goto L_EVAL;
          }
        }

        sub_sp = CONT_SAVED_SP(c0);
        val = mk_tup((Ptr)CONT_PAYLOAD(c0), val);
        cont_sp = cont_sp - 1;
        if (cont_sp == 0) goto L_DONE;
        c0 = cont_stack[cont_sp - 1];
        if (CONT_TAG(c0) != CONT_TUP_DONE) break;
      }
      goto L_RET;
    }

    if (tag == CONT_TUP_SND) {
      sub_sp = CONT_SAVED_SP(c0);
      cont_stack[cont_sp - 1] = CONT_MAKE(CONT_TUP_DONE, 0, sub_sp, val);
      code_id = CONT_CODE_ID(c0);
      cc = (code_id == CODE_PROG) ? PROG_CODE->buf : c->buf;
      pc = CONT_PAYLOAD(c0);
      goto L_EVAL;
    }

    if (tag == CONT_REC_FEED) {
      uint32_t saved_sp = CONT_SAVED_SP(c0);
      sub_sp = 0;
      cont_stack[cont_sp - 1] = CONT_MAKE(CONT_REC_DONE, 0, saved_sp, 0);
      code_id = CODE_PROG;
      cc = PROG_CODE->buf;

      if (top_get_mat) {
        HOT_ASSERT(PTR_TAG(val) == CTR_TUP, "Get expected tuple.");
#if LAMBIT_COUNT_STATS
        app_get++;
#endif
        uint16_t l = PTR_LOC(val);
        Ptr a;
        Ptr b;
        heap_get_pair(l, &a, &b);
        free2(l);
#if LAMBIT_COUNT_STATS
        app_mat++;
#endif
        uint32_t nxt_pc = (a == PTR_BT0) ? (top_pc + 2) : (top_pc + 2 + top_mat_delta);
        pc = feed_term(nxt_pc, b, &sub_sp, &app_lam, &app_mat, &app_get, &app_use);
      } else if (top_get) {
        HOT_ASSERT(PTR_TAG(val) == CTR_TUP, "Get expected tuple.");
#if LAMBIT_COUNT_STATS
        app_get++;
#endif
        uint16_t l = PTR_LOC(val);
        Ptr a;
        Ptr b;
        heap_get_pair(l, &a, &b);
        free2(l);
        pc = feed_term(top_pc + 1, a, &sub_sp, &app_lam, &app_mat, &app_get, &app_use);
        pc = feed_term(pc, b, &sub_sp, &app_lam, &app_mat, &app_get, &app_use);
      } else {
        pc = feed_term(top_pc, val, &sub_sp, &app_lam, &app_mat, &app_get, &app_use);
      }
      goto L_EVAL;
    }

    if (tag == CONT_REC_DONE) {
      sub_sp = CONT_SAVED_SP(c0);
      cont_sp = cont_sp - 1;
      goto L_RET;
    }

    die("Invalid eval continuation tag %u.", (unsigned)tag);
  }

L_BAD: {
    die("Expected term opcode, got 0x%02X at pc=%u.", op, pc);
  }

L_DONE: {
    SUB_SP = sub_sp;
    STATS.app_fun = app_fun;
    STATS.app_lam = app_lam;
    STATS.app_mat = app_mat;
    STATS.app_get = app_get;
    STATS.app_use = app_use;
    return val;
  }

#undef CONT_PUSH
  return PTR_NUL;
}

#undef CONT_PAYLOAD
#undef CONT_SAVED_SP
#undef CONT_CODE_ID
#undef CONT_TAG
#undef CONT_MAKE

// ---------------- Show ----------------

static void show_ptr(Ptr p) {
  switch (PTR_TAG(p)) {
    case CTR_NUL: printf("()"); return;
    case CTR_BT0: printf("0"); return;
    case CTR_BT1: printf("1"); return;
    case CTR_TUP: {
      uint16_t l = PTR_LOC(p);
      printf("(");
      show_ptr(heap_get_fst(l));
      printf(",");
      show_ptr(heap_get_snd(l));
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
  PROG_TOP_GET_MAT = false;
  PROG_TOP_MAT_DELTA = 0;
  if (PROG_TOP_GET && is_mat(prog.buf[PROG_PC + 1])) {
    PROG_TOP_GET_MAT = true;
    PROG_TOP_MAT_DELTA = (uint32_t)(prog.buf[PROG_PC + 1] - OP_MAT);
  }
  uint32_t input_pc = compile_term(&input, input_heap);

  SUB_SP = 0;
  Ptr result = eval_term(&input, input_pc);
  show_ptr(result);
  printf("\n");
  show_stats(&STATS);

  free(input_heap);
  return 0;
}
