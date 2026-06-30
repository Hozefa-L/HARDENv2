contract LandRegistryEscrow {
    mapping(address => uint256) public escrow;
    function depositFee() external payable {
        escrow[msg.sender] += msg.value;
    }
    function refundFailedRegistration() external {
        uint256 amt = escrow[msg.sender];
        require(amt > 0, "no escrow");
        (bool ok, ) = msg.sender.call{value: amt}("");
        require(ok, "send failed");
        escrow[msg.sender] = 0; // state update AFTER call: SWC-107
    }
}