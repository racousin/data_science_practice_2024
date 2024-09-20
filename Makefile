# Makefile

# Define the Python environment activation
VENV_ACTIVATE = . venv/bin/activate

# Find all .ipynb files in the specified paths, excluding checkpoints
NOTEBOOKS := $(shell find website/public/modules/module* -type f -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*')

# Define the HTML targets
HTMLS := $(NOTEBOOKS:.ipynb=.html)

# Default target
all: $(HTMLS)

# Rule to convert .ipynb to .html
%.html: %.ipynb
	@echo "Converting $< to $@"
	@bash -c "$(VENV_ACTIVATE) && jupyter nbconvert --to html \"$<\""

.PHONY: clean

# Clean generated HTML files
clean:
	@echo "Cleaning up HTML files..."
	@rm -f $(HTMLS)
