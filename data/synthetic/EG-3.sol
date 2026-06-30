contract TitleTransferOrdering {
    address public registrar;
    uint256 public feeRate;             // mutable, mempool-visible
    mapping(uint256 => address) public title;
    function setFee(uint256 f) external {
        require(msg.sender == registrar);
        feeRate = f;
    }
    function transferTitle(uint256 id, address to) external payable {
        require(msg.value >= feeRate);    // SWC-114: order-sensitive
        title[id] = to;
    }
}