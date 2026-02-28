// LamBit
// ======
// A minimal functional language based on binary pattern-matching case-trees.
//
//     Func ::=
//       | Lam ::= "λ" Name "." Func
//       | Mat ::= "λ" "{" "0" ":" Func ";" "1" ":" Func ";" "}"
//       | Use ::= "λ" "()" "." Func
//       | Get ::= "λ" "!" Func
//       | Ret ::= Term
//
//     Term ::=
//       | Var ::= Name
//       | Bt0 ::= "0"
//       | Bt1 ::= "1"
//       | Nul ::= "()"
//       | Tup ::= "(" Term "," Term ")"
//       | Rec ::= "~" Term
//
// LamBit programs are composed by a single "top-level" function. Computation
// happens by recursively pattern-matching that function against a single input.
// 
// LamBit is Turing-complete. For example, consider the following Haskell program:
// 
//     data Nat    = Z | S Nat
//     add Z     b = b
//     add (S a) b = S (add a b)
//     mul Z     b = Z
//     mul (S a) b = add b (mul a b)
// 
// It can be expressed on LamBit as:
//
//     F = λ! λ{
//       0: λ! λ{ // computes addition
//         0: λ(). b
//         1: λa. λb. (1,~(0,(a,b)))
//       }
//       1: λ! λ{ // computes multiplication
//         0: λb. 0{}
//         1: λa. λb. ~(0,(b,~(1,(a,b))))
//       }
//     }
//
// Where natural numbers are encoded as:
//
//     Nat ::=
//     | zero ::= (0,())
//     | succ ::= (1,Nat)
//     (aliasing 3n as (1,(1,(1,(0,())))), and so on)
//
// For instance, ~(1,(2n,3n)) evaluates 2n*3n to 6n:
//
//     ~(1,(2n,3n))
//     = ~(0,(3n, ~(1,(1n,3n))))
//     = ~(0,(3n, ~(0,(3n, ~(1,((0,()),3n))))))
//     = ~(0,(3n, ~(0,(3n, (0,())))))
//     = ~(0,(3n, 3n))
//     = (1,~(0,(2n,3n)))
//     = (1,(1,~(0,(1n,3n))))
//     = (1,(1,(1,~(0,((0,()),3n)))))
//     = (1,(1,(1,3n)))
//     = 6n

#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Pointers
#define PTR_TUP 0x0000
#define PTR_NUL 0x4000
#define PTR_BT0 0x8000
#define PTR_BT1 0xC000

// Memory and GC: Dense packed 32-bit heap & O(1) freelist allocator
alignas(64) uint32_t heap[16384]; 
uint32_t freelist[16384];
uint32_t free_sp = 0;

// Allocation profiling counters
uint64_t alloc_calls = 0;
uint64_t free_calls = 0;
uint32_t max_K_seen = 0;

void init_heap() {
    free_sp = 16383;
    for (uint32_t i = 0; i < 16383; i++) {
        freelist[i] = 16383 - i; 
    }
}

// Runtime State
typedef struct {
    uint32_t app_fun, app_lam, app_mat, app_get, app_use;
} Stats;

enum {
    OP_DONE = 0,
    OP_GET, OP_LAM, OP_USE, OP_MAT,
    OP_NUL, OP_BT0, OP_BT1, OP_VAR, OP_TUP, OP_REC, OP_TAIL_REC, OP_END,
    OP_GET_MAT, OP_GET_LAM, OP_GET_USE, OP_REC_TUP, OP_TAIL_REC_TUP,
    OP_VAR_TUP, OP_BT0_TUP, OP_BT1_TUP, OP_NUL_TUP,
    OP_VAR_TUP_REC, OP_VAR_TUP_TAIL_REC,
    OP_BT0_TUP_REC, OP_BT0_TUP_TAIL_REC,
    OP_BT1_TUP_REC, OP_BT1_TUP_TAIL_REC,
    OP_NUL_TUP_REC, OP_NUL_TUP_TAIL_REC
};

uint16_t prog[4096];
uint32_t prog_len = 0;
Stats stats = {0};

char names[24][16];
int name_count = 0;

int find_name(const char* name) {
    for (int i = name_count - 1; i >= 0; i--) {
        if (strcmp(names[i], name) == 0) return i;
    }
    return -1;
}

// Parser
void skip_ws(const char* src, uint32_t* idx) {
    while (src[*idx] == ' ' || src[*idx] == '\n' || src[*idx] == '\t' || src[*idx] == '\r') (*idx)++;
}

bool match_str(const char* src, uint32_t* idx, const char* str) {
    skip_ws(src, idx);
    uint32_t len = strlen(str);
    if (strncmp(src + *idx, str, len) == 0) {
        *idx += len;
        return true;
    }
    return false;
}

void expect_str(const char* src, uint32_t* idx, const char* str) {
    if (!match_str(src, idx, str)) {
        printf("Expected '%s' at index %u\n", str, *idx);
        exit(1);
    }
}

void parse_name_str(const char* src, uint32_t* idx, char* out) {
    skip_ws(src, idx);
    int len = 0;
    while (src[*idx] && strchr("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789", src[*idx])) {
        out[len++] = src[(*idx)++];
    }
    out[len] = '\0';
}

bool is_bt(const char* src, uint32_t* idx, char c) {
    skip_ws(src, idx);
    if (src[*idx] == c) {
        char next = src[*idx+1];
        if ((next >= 'a' && next <= 'z') || (next >= 'A' && next <= 'Z') || (next >= '0' && next <= '9') || next == '_' || next == '{') {
            return false;
        }
        return true;
    }
    return false;
}

void parse_term_src(const char* src, uint32_t* idx);

void parse_func_src(const char* src, uint32_t* idx) {
    skip_ws(src, idx);
    if (match_str(src, idx, "λ!")) {
        prog[prog_len++] = OP_GET;
        parse_func_src(src, idx);
        return;
    }
    if (match_str(src, idx, "λ")) {
        if (match_str(src, idx, "{")) {
            expect_str(src, idx, "0"); expect_str(src, idx, ":");
            uint32_t op_pos = prog_len;
            prog[prog_len++] = OP_MAT;
            prog[prog_len++] = 0; 
            parse_func_src(src, idx);
            match_str(src, idx, ";");
            expect_str(src, idx, "1"); expect_str(src, idx, ":");
            prog[op_pos + 1] = prog_len - (op_pos + 2);
            parse_func_src(src, idx);
            match_str(src, idx, ";");
            expect_str(src, idx, "}");
            return;
        }
        if (match_str(src, idx, "()")) {
            expect_str(src, idx, ".");
            prog[prog_len++] = OP_USE;
            parse_func_src(src, idx);
            return;
        }
        char name[16];
        parse_name_str(src, idx, name);
        expect_str(src, idx, ".");
        strcpy(names[name_count++], name);
        prog[prog_len++] = OP_LAM;
        parse_func_src(src, idx);
        name_count--;
        return;
    }
    
    skip_ws(src, idx);
    if (match_str(src, idx, "~")) {
        parse_term_src(src, idx);
        prog[prog_len++] = OP_TAIL_REC;
    } else {
        parse_term_src(src, idx);
        prog[prog_len++] = OP_END;
    }
}

void parse_term_src(const char* src, uint32_t* idx) {
    skip_ws(src, idx);
    if (match_str(src, idx, "~")) {
        parse_term_src(src, idx);
        prog[prog_len++] = OP_REC;
        return;
    }
    if (src[*idx] == '(' && src[*idx+1] == ')') {
        *idx += 2;
        prog[prog_len++] = OP_NUL;
        return;
    }
    if (src[*idx] == '(') {
        match_str(src, idx, "(");
        parse_term_src(src, idx); 
        expect_str(src, idx, ",");
        parse_term_src(src, idx); 
        expect_str(src, idx, ")");
        prog[prog_len++] = OP_TUP;
        return;
    }
    if (is_bt(src, idx, '0')) {
        (*idx)++;
        prog[prog_len++] = OP_BT0;
        return;
    }
    if (is_bt(src, idx, '1')) {
        (*idx)++;
        prog[prog_len++] = OP_BT1;
        return;
    }
    char name[16];
    parse_name_str(src, idx, name);
    int var_idx = find_name(name);
    if (var_idx == -1) {
        printf("Unbound variable '%s'\n", name);
        exit(1);
    }
    prog[prog_len++] = OP_VAR;
    prog[prog_len++] = var_idx;
}

static void** dispatch_table;
static void* compiled_prog[4096];
static uint32_t pc_map[4096];

static uint32_t val_stack[8192];
static uint32_t term_stack[8192];
static void* call_stack_pc[8192];
static uint32_t* call_stack_ctx[16384];
static uint32_t ctx[1024 * 64];

static uint32_t eval_app_from(void** pc_start, void** prog_start) {
    if (__builtin_expect(prog_start == NULL, 0)) {
        static const void* dispatch[] = {
            &&L_END,         // OP_DONE (0)
            &&L_GET,         // OP_GET
            &&L_LAM,         // OP_LAM
            &&L_USE,         // OP_USE
            &&L_MAT,         // OP_MAT
            &&L_NUL,         // OP_NUL
            &&L_BT0,         // OP_BT0
            &&L_BT1,         // OP_BT1
            &&L_VAR,         // OP_VAR
            &&L_TUP,         // OP_TUP
            &&L_REC,         // OP_REC
            &&L_TAIL_REC,    // OP_TAIL_REC
            &&L_END,         // OP_END
            &&L_GET_MAT,     // OP_GET_MAT
            &&L_GET_LAM,     // OP_GET_LAM
            &&L_GET_USE,     // OP_GET_USE
            &&L_REC_TUP,     // OP_REC_TUP
            &&L_TAIL_REC_TUP,// OP_TAIL_REC_TUP
            &&L_VAR_TUP,     // OP_VAR_TUP
            &&L_BT0_TUP,     // OP_BT0_TUP
            &&L_BT1_TUP,     // OP_BT1_TUP
            &&L_NUL_TUP,     // OP_NUL_TUP
            &&L_VAR_TUP_REC, // OP_VAR_TUP_REC
            &&L_VAR_TUP_TAIL_REC, // OP_VAR_TUP_TAIL_REC
            &&L_BT0_TUP_REC, &&L_BT0_TUP_TAIL_REC,
            &&L_BT1_TUP_REC, &&L_BT1_TUP_TAIL_REC,
            &&L_NUL_TUP_REC, &&L_NUL_TUP_TAIL_REC
        };
        dispatch_table = (void**)dispatch;
        return 0;
    }

    uint32_t* restrict v_sp = val_stack;
    uint32_t* restrict t_sp = term_stack;
    void*** restrict c_sp_pc = (void***)call_stack_pc;
    uint32_t** restrict c_sp_ctx = call_stack_ctx;
    
    uint32_t* restrict current_ctx = ctx;
    uint32_t* restrict frame_base = ctx;

    void** restrict pc = pc_start;
    uint32_t term = 0;
    uint32_t arg = 0;

    uint32_t* restrict l_freelist = freelist;
    uint32_t* restrict l_heap = heap;

    uint32_t l_free_sp = free_sp;
    uint32_t min_free_sp = l_free_sp;
    
    uint64_t l_alloc_calls = 0;
    uint64_t l_free_calls = 0;

    uint32_t l_app_fun = 0;
    uint32_t l_app_lam = 0;
    uint32_t l_app_mat = 0;
    uint32_t l_app_get = 0;
    uint32_t l_app_use = 0;

#define ALLOC(fst, snd, out) do { \
    l_alloc_calls++; \
    out = l_freelist[--l_free_sp]; \
    if (l_free_sp < min_free_sp) min_free_sp = l_free_sp; \
    l_heap[out] = (fst) | ((snd) << 16); \
} while(0)

#define FREE_LOC(loc) do { \
    l_free_calls++; \
    l_freelist[l_free_sp++] = (loc); \
} while(0)

    goto **pc++;

tail_call:
    pc = prog_start;
    term = arg;
    t_sp = term_stack; 
    goto **pc++;

L_GET:
    l_app_get++;
    {
        uint32_t val = l_heap[term];
        FREE_LOC(term);
        *t_sp++ = val >> 16;
        term = val & 0xFFFF;
    }
    goto **pc++;

L_LAM:
    l_app_lam++;
    *current_ctx++ = term;
    if (t_sp > term_stack) term = *--t_sp;
    goto **pc++;

L_USE:
    l_app_use++;
    if (t_sp > term_stack) term = *--t_sp;
    goto **pc++;

L_MAT:
    l_app_mat++;
    {
        void** branch1 = (void**)*pc++;
        if (term == PTR_BT1) pc = branch1;
    }
    if (t_sp > term_stack) term = *--t_sp;
    goto **pc++;

L_NUL: 
    *v_sp++ = PTR_NUL; 
    goto **pc++;

L_BT0: 
    *v_sp++ = PTR_BT0; 
    goto **pc++;

L_BT1: 
    *v_sp++ = PTR_BT1; 
    goto **pc++;

L_VAR:
    *v_sp++ = frame_base[(uintptr_t)(*pc++)];
    goto **pc++;

L_TUP:
    {
        uint32_t right = *--v_sp;
        uint32_t left  = *(v_sp - 1);
        ALLOC(left, right, *(v_sp - 1));
    }
    goto **pc++;

L_VAR_TUP:
    {
        uint32_t right = frame_base[(uintptr_t)(*pc++)];
        uint32_t left  = *(v_sp - 1);
        ALLOC(left, right, *(v_sp - 1));
    }
    goto **pc++;

L_BT0_TUP:
    {
        uint32_t left  = *(v_sp - 1);
        ALLOC(left, PTR_BT0, *(v_sp - 1));
    }
    goto **pc++;

L_BT1_TUP:
    {
        uint32_t left  = *(v_sp - 1);
        ALLOC(left, PTR_BT1, *(v_sp - 1));
    }
    goto **pc++;

L_NUL_TUP:
    {
        uint32_t left  = *(v_sp - 1);
        ALLOC(left, PTR_NUL, *(v_sp - 1));
    }
    goto **pc++;

L_REC:
    l_app_fun++;
    {
        arg = *--v_sp;
        *c_sp_pc++ = pc;
        *c_sp_ctx++ = frame_base;
        *c_sp_ctx++ = current_ctx;
        frame_base = current_ctx;
    }
    goto tail_call;

L_TAIL_REC:
    l_app_fun++;
    arg = *--v_sp;
    current_ctx = frame_base; 
    goto tail_call;

L_END:
    {
        uint32_t res = *--v_sp;
        if (c_sp_pc == (void***)call_stack_pc) {
            stats.app_fun += l_app_fun;
            stats.app_lam += l_app_lam;
            stats.app_mat += l_app_mat;
            stats.app_get += l_app_get;
            stats.app_use += l_app_use;
            alloc_calls += l_alloc_calls;
            free_calls += l_free_calls;
            free_sp = l_free_sp;
            uint32_t used = 16383 - min_free_sp;
            if (used > max_K_seen) max_K_seen = used;
            return res;
        }
        
        current_ctx = *--c_sp_ctx;
        frame_base = *--c_sp_ctx;
        pc = *--c_sp_pc;
        
        *v_sp++ = res;
    }
    goto **pc++;

L_GET_MAT:
    l_app_get++;
    l_app_mat++;
    {
        void** branch1 = (void**)*pc++;
        uint32_t val = l_heap[term];
        FREE_LOC(term);
        if ((val & 0xFFFF) == PTR_BT1) pc = branch1;
        term = val >> 16;
    }
    goto **pc++;

L_GET_LAM:
    l_app_get++;
    l_app_lam++;
    {
        uint32_t val = l_heap[term];
        FREE_LOC(term);
        *current_ctx++ = val & 0xFFFF;
        term = val >> 16;
    }
    goto **pc++;

L_GET_USE:
    l_app_get++;
    l_app_use++;
    {
        uint32_t val = l_heap[term];
        FREE_LOC(term);
        term = val >> 16;
    }
    goto **pc++;

L_REC_TUP:
    l_app_fun++;
    l_app_get++;
    {
        uint32_t right = *--v_sp;
        uint32_t left  = *--v_sp;
        
        *c_sp_pc++ = pc;
        *c_sp_ctx++ = frame_base;
        *c_sp_ctx++ = current_ctx;
        
        frame_base = current_ctx;
        
        t_sp = term_stack;
        *t_sp++ = right;
        term = left;
        pc = prog_start + 1;
    }
    goto **pc++;

L_TAIL_REC_TUP:
    l_app_fun++;
    l_app_get++;
    {
        uint32_t right = *--v_sp;
        uint32_t left  = *--v_sp;
        
        t_sp = term_stack;
        *t_sp++ = right;
        term = left;
        
        current_ctx = frame_base;
        pc = prog_start + 1;
    }
    goto **pc++;

L_VAR_TUP_REC:
    l_app_fun++;
    l_app_get++;
    {
        uint32_t right = frame_base[(uintptr_t)(*pc++)];
        uint32_t left  = *--v_sp;
        
        *c_sp_pc++ = pc;
        *c_sp_ctx++ = frame_base;
        *c_sp_ctx++ = current_ctx;
        
        frame_base = current_ctx;
        
        t_sp = term_stack;
        *t_sp++ = right;
        term = left;
        pc = prog_start + 1;
    }
    goto **pc++;

L_VAR_TUP_TAIL_REC:
    l_app_fun++;
    l_app_get++;
    {
        uint32_t right = frame_base[(uintptr_t)(*pc++)];
        uint32_t left  = *--v_sp;
        
        t_sp = term_stack;
        *t_sp++ = right;
        term = left;
        
        current_ctx = frame_base;
        pc = prog_start + 1;
    }
    goto **pc++;

L_BT0_TUP_REC:
    l_app_fun++; l_app_get++;
    {
        uint32_t left = *--v_sp;
        *c_sp_pc++ = pc; *c_sp_ctx++ = frame_base; *c_sp_ctx++ = current_ctx;
        frame_base = current_ctx;
        t_sp = term_stack; *t_sp++ = PTR_BT0; term = left; pc = prog_start + 1;
    }
    goto **pc++;

L_BT0_TUP_TAIL_REC:
    l_app_fun++; l_app_get++;
    {
        uint32_t left = *--v_sp;
        t_sp = term_stack; *t_sp++ = PTR_BT0; term = left; current_ctx = frame_base; pc = prog_start + 1;
    }
    goto **pc++;

L_BT1_TUP_REC:
    l_app_fun++; l_app_get++;
    {
        uint32_t left = *--v_sp;
        *c_sp_pc++ = pc; *c_sp_ctx++ = frame_base; *c_sp_ctx++ = current_ctx;
        frame_base = current_ctx;
        t_sp = term_stack; *t_sp++ = PTR_BT1; term = left; pc = prog_start + 1;
    }
    goto **pc++;

L_BT1_TUP_TAIL_REC:
    l_app_fun++; l_app_get++;
    {
        uint32_t left = *--v_sp;
        t_sp = term_stack; *t_sp++ = PTR_BT1; term = left; current_ctx = frame_base; pc = prog_start + 1;
    }
    goto **pc++;

L_NUL_TUP_REC:
    l_app_fun++; l_app_get++;
    {
        uint32_t left = *--v_sp;
        *c_sp_pc++ = pc; *c_sp_ctx++ = frame_base; *c_sp_ctx++ = current_ctx;
        frame_base = current_ctx;
        t_sp = term_stack; *t_sp++ = PTR_NUL; term = left; pc = prog_start + 1;
    }
    goto **pc++;

L_NUL_TUP_TAIL_REC:
    l_app_fun++; l_app_get++;
    {
        uint32_t left = *--v_sp;
        t_sp = term_stack; *t_sp++ = PTR_NUL; term = left; current_ctx = frame_base; pc = prog_start + 1;
    }
    goto **pc++;
}

void compile_prog() {
    uint32_t j = 0;
    for (uint32_t i = 0; i < prog_len; i++) {
        pc_map[i] = j;
        uint16_t op = prog[i];
        
        if (op == OP_GET && i > 0 && i + 1 < prog_len && prog[i+1] == OP_MAT) {
            j += 2; i += 2;
        } else if (op == OP_GET && i > 0 && i + 1 < prog_len && prog[i+1] == OP_LAM) {
            j += 1; i += 1;
        } else if (op == OP_GET && i > 0 && i + 1 < prog_len && prog[i+1] == OP_USE) {
            j += 1; i += 1;
        } else if (op == OP_VAR && i + 2 < prog_len && prog[i+2] == OP_TUP) {
            if (i + 3 < prog_len && prog[i+3] == OP_REC) { j += 2; i += 3; }
            else if (i + 3 < prog_len && prog[i+3] == OP_TAIL_REC) { j += 2; i += 3; }
            else { j += 2; i += 2; }
        } else if (op == OP_BT0 && i + 1 < prog_len && prog[i+1] == OP_TUP) {
            if (i + 2 < prog_len && prog[i+2] == OP_REC) { j += 1; i += 2; }
            else if (i + 2 < prog_len && prog[i+2] == OP_TAIL_REC) { j += 1; i += 2; }
            else { j += 1; i += 1; }
        } else if (op == OP_BT1 && i + 1 < prog_len && prog[i+1] == OP_TUP) {
            if (i + 2 < prog_len && prog[i+2] == OP_REC) { j += 1; i += 2; }
            else if (i + 2 < prog_len && prog[i+2] == OP_TAIL_REC) { j += 1; i += 2; }
            else { j += 1; i += 1; }
        } else if (op == OP_NUL && i + 1 < prog_len && prog[i+1] == OP_TUP) {
            if (i + 2 < prog_len && prog[i+2] == OP_REC) { j += 1; i += 2; }
            else if (i + 2 < prog_len && prog[i+2] == OP_TAIL_REC) { j += 1; i += 2; }
            else { j += 1; i += 1; }
        } else if (op == OP_TUP && i + 1 < prog_len && prog[i+1] == OP_REC) {
            j += 1; i += 1;
        } else if (op == OP_TUP && i + 1 < prog_len && prog[i+1] == OP_TAIL_REC) {
            j += 1; i += 1;
        } else if (op == OP_MAT) {
            j += 2; i += 1;
        } else if (op == OP_VAR) {
            j += 2; i += 1;
        } else {
            j += 1;
        }
    }
    
    j = 0;
    for (uint32_t i = 0; i < prog_len; i++) {
        uint16_t op = prog[i];
        
        if (op == OP_GET && i > 0 && i + 1 < prog_len && prog[i+1] == OP_MAT) {
            compiled_prog[j++] = dispatch_table[OP_GET_MAT];
            compiled_prog[j++] = &compiled_prog[pc_map[i + 3 + prog[i+2]]];
            i += 2;
        } else if (op == OP_GET && i > 0 && i + 1 < prog_len && prog[i+1] == OP_LAM) {
            compiled_prog[j++] = dispatch_table[OP_GET_LAM];
            i += 1;
        } else if (op == OP_GET && i > 0 && i + 1 < prog_len && prog[i+1] == OP_USE) {
            compiled_prog[j++] = dispatch_table[OP_GET_USE];
            i += 1;
        } else if (op == OP_VAR && i + 2 < prog_len && prog[i+2] == OP_TUP) {
            if (i + 3 < prog_len && prog[i+3] == OP_REC) {
                compiled_prog[j++] = dispatch_table[OP_VAR_TUP_REC];
                compiled_prog[j++] = (void*)(uintptr_t)prog[i+1];
                i += 3;
            } else if (i + 3 < prog_len && prog[i+3] == OP_TAIL_REC) {
                compiled_prog[j++] = dispatch_table[OP_VAR_TUP_TAIL_REC];
                compiled_prog[j++] = (void*)(uintptr_t)prog[i+1];
                i += 3;
            } else {
                compiled_prog[j++] = dispatch_table[OP_VAR_TUP];
                compiled_prog[j++] = (void*)(uintptr_t)prog[i+1];
                i += 2;
            }
        } else if (op == OP_BT0 && i + 1 < prog_len && prog[i+1] == OP_TUP) {
            if (i + 2 < prog_len && prog[i+2] == OP_REC) { compiled_prog[j++] = dispatch_table[OP_BT0_TUP_REC]; i += 2; }
            else if (i + 2 < prog_len && prog[i+2] == OP_TAIL_REC) { compiled_prog[j++] = dispatch_table[OP_BT0_TUP_TAIL_REC]; i += 2; }
            else { compiled_prog[j++] = dispatch_table[OP_BT0_TUP]; i += 1; }
        } else if (op == OP_BT1 && i + 1 < prog_len && prog[i+1] == OP_TUP) {
            if (i + 2 < prog_len && prog[i+2] == OP_REC) { compiled_prog[j++] = dispatch_table[OP_BT1_TUP_REC]; i += 2; }
            else if (i + 2 < prog_len && prog[i+2] == OP_TAIL_REC) { compiled_prog[j++] = dispatch_table[OP_BT1_TUP_TAIL_REC]; i += 2; }
            else { compiled_prog[j++] = dispatch_table[OP_BT1_TUP]; i += 1; }
        } else if (op == OP_NUL && i + 1 < prog_len && prog[i+1] == OP_TUP) {
            if (i + 2 < prog_len && prog[i+2] == OP_REC) { compiled_prog[j++] = dispatch_table[OP_NUL_TUP_REC]; i += 2; }
            else if (i + 2 < prog_len && prog[i+2] == OP_TAIL_REC) { compiled_prog[j++] = dispatch_table[OP_NUL_TUP_TAIL_REC]; i += 2; }
            else { compiled_prog[j++] = dispatch_table[OP_NUL_TUP]; i += 1; }
        } else if (op == OP_TUP && i + 1 < prog_len && prog[i+1] == OP_REC) {
            compiled_prog[j++] = dispatch_table[OP_REC_TUP];
            i += 1;
        } else if (op == OP_TUP && i + 1 < prog_len && prog[i+1] == OP_TAIL_REC) {
            compiled_prog[j++] = dispatch_table[OP_TAIL_REC_TUP];
            i += 1;
        } else if (op == OP_MAT) {
            compiled_prog[j++] = dispatch_table[OP_MAT];
            compiled_prog[j++] = &compiled_prog[pc_map[i + 2 + prog[i+1]]];
            i += 1;
        } else if (op == OP_VAR) {
            compiled_prog[j++] = dispatch_table[OP_VAR];
            compiled_prog[j++] = (void*)(uintptr_t)prog[i+1];
            i += 1;
        } else {
            compiled_prog[j++] = dispatch_table[op];
        }
    }
}

void show_term(uint32_t ptr) {
    if (ptr == PTR_NUL) {
        printf("()");
    } else if (ptr == PTR_BT0) {
        printf("0");
    } else if (ptr == PTR_BT1) {
        printf("1");
    } else {
        printf("(");
        uint32_t val = heap[ptr];
        show_term(val & 0xFFFF);
        printf(",");
        show_term(val >> 16);
        printf(")");
    }
}


int main() {
    init_heap();

    const char* prog_src = "λ! λ{\n"
    "    0: λ! λ{\n"
    "      0: λx. ~(0,(1,(~(1,(1,x)),(0,()))))\n"
    "      1: λ! λ! λ{\n"
    "        0: λ(). λzs. zs\n"
    "        1: λ! λ{\n"
    "          0: λxs. λzs. ~(0,(1,(xs,(1,(0,zs)))))\n"
    "          1: λxs. λzs. ~(0,(0,~(1,(0,(zs,(1,(1,xs)))))))\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "    1: λ! λ{\n"
    "      0: λ! λ! λ{\n"
    "        0: λ(). λys. ys\n"
    "        1: λ! λx. λxs. λys. (1,(x,~(1,(0,(xs,ys)))))\n"
    "      }\n"
    "      1: λ! λ{\n"
    "        0: λ(). (0,())\n"
    "        1: λ! λ{\n"
    "          0: λxs. (1,(1,~(1,(1,xs))))\n"
    "          1: λxs. (1,(0,xs))\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }";

    uint32_t idx = 0;
    parse_func_src(prog_src, &idx);

    char n_buf[2048];
    strcpy(n_buf, "(0,())");
    for (int i = 0; i < 24; i++) {
        char temp[2048];
        sprintf(temp, "(1,(1,%s))", n_buf);
        strcpy(n_buf, temp);
    }
    char input_src[2048];
    sprintf(input_src, "~(0,(0,%s))", n_buf);

    uint32_t input_idx = 0;
    uint32_t input_pc = prog_len;
    parse_term_src(input_src, &input_idx);
    prog[prog_len++] = OP_END;

    eval_app_from(NULL, NULL);
    compile_prog();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint32_t result = eval_app_from(&compiled_prog[pc_map[input_pc]], compiled_prog);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    show_term(result);
    printf("\n");
    uint32_t total = stats.app_fun + stats.app_lam + stats.app_mat + stats.app_get + stats.app_use;
    printf("Interactions: %u\n", total);
    printf("- APP-FUN: %u\n", stats.app_fun);
    printf("- APP-LAM: %u\n", stats.app_lam);
    printf("- APP-MAT: %u\n", stats.app_mat);
    printf("- APP-GET: %u\n", stats.app_get);
    printf("- APP-USE: %u\n", stats.app_use);

    printf("\nTime: %.4f seconds\n", elapsed);
    printf("Interactions/s: %.0f\n", elapsed > 0 ? (double)total / elapsed : 0.0);

    printf("\nAllocation profiling:\n");
    printf("- alloc calls:      %llu\n", (unsigned long long)alloc_calls);
    printf("- alloc scans:      0\n");
    printf("- scans/alloc:      0.0000\n");
    printf("- free_term calls:  %llu\n", (unsigned long long)free_calls);
    printf("- max K seen:       %u (of 16384 slots = %.2f%% fill)\n", max_K_seen, (double)max_K_seen / 16384.0 * 100.0);

    return 0;
}
