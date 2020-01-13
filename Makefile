CONTAINERS=syk.sif base.sif
SINGULARITY=sudo singularity

.PHONY: all
all: syk.sif

.PHONY: clean
clean:
	rm -f ${CONTAINERS}

%.sif: %.def
	$(SINGULARITY) build --force $@ $<

syk.sif: base.sif $(shell find src -type f)
