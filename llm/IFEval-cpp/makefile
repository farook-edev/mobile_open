CXX         ?= g++
AR          ?= ar
CXXFLAGSLIB ?= -O2 -std=c++11 -fPIC
CXXFLAGS    ?= -O2 -std=c++17 -fPIC -Ilibcld2
LDFLAGS     ?=
PKG_LIBS    := -Llibcld2 -lstatcld2

# Object files that make up libstatcld2
LIBCLD2_OBJS = \
	libcld2/internal/cldutil.o \
	libcld2/internal/cldutil_shared.o \
	libcld2/internal/compact_lang_det.o \
	libcld2/internal/compact_lang_det_hint_code.o \
	libcld2/internal/compact_lang_det_impl.o \
	libcld2/internal/debug.o \
	libcld2/internal/fixunicodevalue.o \
	libcld2/internal/generated_entities.o \
	libcld2/internal/generated_language.o \
	libcld2/internal/generated_ulscript.o \
	libcld2/internal/getonescriptspan.o \
	libcld2/internal/lang_script.o \
	libcld2/internal/offsetmap.o \
	libcld2/internal/scoreonescriptspan.o \
	libcld2/internal/tote.o \
	libcld2/internal/utf8statetable.o \
	libcld2/internal/cld_generated_cjk_uni_prop_80.o \
	libcld2/internal/cld2_generated_cjk_compatible.o \
	libcld2/internal/cld_generated_cjk_delta_bi_4.o \
	libcld2/internal/generated_distinct_bi_0.o \
	libcld2/internal/cld2_generated_quadchrome_2.o \
	libcld2/internal/cld2_generated_deltaoctachrome.o \
	libcld2/internal/cld2_generated_distinctoctachrome.o \
	libcld2/internal/cld_generated_score_quad_octa_2.o

STATLIB = libcld2/libstatcld2.a
SHLIB   = libcld2/libstatcld2.so

.PHONY: all clean

all: main

main: ifeval.o main.o $(STATLIB)
	$(CXX) $(CXXFLAGS) -o $@ ifeval.o main.o $(STATLIB) $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

ifeval.o: ifeval.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SHLIB): $(STATLIB)
	@mkdir -p $(dir $@)
	$(CXX) -shared -o $@ -Wl,--whole-archive $^ -Wl,--no-whole-archive $(LDFLAGS)

$(STATLIB): $(LIBCLD2_OBJS)
	@mkdir -p $(dir $@)
	$(AR) crs $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGSLIB) -c $< -o $@

clean:
	$(RM) $(LIBCLD2_OBJS) $(STATLIB) $(SHLIB) main.o main
