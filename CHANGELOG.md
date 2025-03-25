# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-03-XX

### Added
- OpenAI-compatible API server using LitServe
- New endpoints:
  - `/v1/audio/speech` for text-to-speech generation
  - `/v1/audio/speech/clone` for voice cloning
- Voice cloning functionality in generator
- API documentation
- Python client examples

### Changed
- Updated requirements.txt with new dependencies
- Enhanced generator class with voice cloning support

## [1.0.0] - Initial Release

### Added
- Initial CSM-1B model implementation
- Gradio web interface
- CLI interface
- Basic text-to-speech functionality
- Multi-speaker support

## [0.0.1] - 2025-03-13

### Added
- Initial release of CSM 1B model
- Basic PyTorch implementation 